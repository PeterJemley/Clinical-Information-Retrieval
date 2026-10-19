"""
Stage 1 of the modulated-composition test: ../PREREGISTRATION.md, Amendments 1 and 2.

Subcommands
  prepare-corpus    PMC Open Access NXML (.tar.gz files or directories) -> corpus.jsonl (+ .meta.json)
  prepare-pubtypes  MEDLINE publication types for the corpus's PMIDs, via NCBI E-utilities -> pubtypes.jsonl
  run               the preregistered analysis, from a JSON configuration (see CONFIG_EXAMPLE.json)

Order inside `run` (Amendment 2, B3): fidelity gates; data checksums; every ranking that needs
no judgment (B0, F, M, U, V, N, R), saved as run files; the B2 manipulation check, saved; only
then are judgments read, L cross-fitted, P scored and the decision rule applied.

check_stage1.py tests every branch of the decision rule, the parsers, the sparse BM25 and a
small end-to-end run on fixtures (B4). The real run refuses the stand-in encoder.
"""
import argparse, array, csv, hashlib, json, math, os, platform, re, subprocess, sys, tarfile, time
import urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import minimize

from clinical_ir.relevance import Document, Query, clinical_relevance, cosine_similarity, actionability_relevance
from clinical_ir.retrieval import BM25
from clinical_ir.evaluate import ndcg_at_k
from clinical_ir.evidence import EvidenceLevel, evidence_weight
from clinical_ir.temporal import temporal_decay

HERE = os.path.dirname(os.path.abspath(__file__))
K, POOL = 10, 100
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = N_PERM = 10_000
LAMBDAS = [0.001, 0.01, 0.1, 1, 10]          # Amendment 2, B1
MAX_PAIRS = 500
MANIPULATION_MIN_SHARE = 0.5                   # Amendment 2, B2
DATE_COVERAGE_MIN = 0.95                       # criterion 1
AUGUST_BM25_NDCG = 0.3103001666796228          # A6 gate target on NFCorpus test

MODULES = ["bm25", "topical", "evidence", "temporal", "applicability", "actionability"]
GATES = {  # registered gate table
    "treatment":      dict(bm25=1, topical=1, evidence=1, temporal=1, applicability=1, actionability=1),
    "diagnosis_test": dict(bm25=1, topical=1, evidence=0, temporal=1, applicability=1, actionability=1),
    "prognosis":      dict(bm25=1, topical=1, evidence=0, temporal=1, applicability=1, actionability=0),
    "other":          dict(bm25=1, topical=1, evidence=0, temporal=0, applicability=0, actionability=0),
}
TYPES = list(GATES)
TOPIC_TYPE_TO_COLUMN = {"diagnosis": "diagnosis_test", "test": "diagnosis_test", "treatment": "treatment"}  # A4
ENCODERS = {
    "medcpt_query": ("ncbi/MedCPT-Query-Encoder", "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc", "cls", 64),
    "medcpt_article": ("ncbi/MedCPT-Article-Encoder", "d05a736da4bb84ee4057b7f7999485be6ed85465", "cls", 512),
    "pubmedbert": ("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                   "e1354b7a3a09615f6aba48dfad4b7a613eef7062", "mean", 512),
}

# ---------- A5: MEDLINE publication type -> evidence level ----------
PT_LEVEL = {
    "Systematic Review": EvidenceLevel.SYSTEMATIC_REVIEW, "Meta-Analysis": EvidenceLevel.SYSTEMATIC_REVIEW,
    "Randomized Controlled Trial": EvidenceLevel.RCT,
    "Controlled Clinical Trial": EvidenceLevel.CONTROLLED_OBSERVATIONAL,
    "Clinical Trial": EvidenceLevel.CONTROLLED_OBSERVATIONAL,
    "Observational Study": EvidenceLevel.CONTROLLED_OBSERVATIONAL,
    "Comparative Study": EvidenceLevel.CONTROLLED_OBSERVATIONAL,
    "Case Reports": EvidenceLevel.UNCONTROLLED_OBSERVATIONAL,
    "Editorial": EvidenceLevel.EXPERT_OPINION, "Comment": EvidenceLevel.EXPERT_OPINION,
    "Letter": EvidenceLevel.EXPERT_OPINION, "Consensus Development Conference": EvidenceLevel.EXPERT_OPINION,
}


def pt_to_level(pub_types):
    levels = [PT_LEVEL[p] for p in (pub_types or []) if p in PT_LEVEL]
    return min(levels) if levels else None  # IntEnum: a lower value is a higher level


def log(msg, events=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if events is not None:
        events.append(line)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- parsing (all inputs are untrusted downloads: no DTDs, no entity declarations) ----------
_DOCTYPE = re.compile(r"<!DOCTYPE[^\[>]*(\[.*?\])?\s*>", re.S)
_NAMED_ENTITY = re.compile(r"&(?!(?:amp|lt|gt|quot|apos);|#\d+;|#x[0-9A-Fa-f]+;)[A-Za-z][A-Za-z0-9.]*;")


def safe_xml(text):
    """Parse XML with the DOCTYPE removed (so no entity can be declared or expanded) and
    undeclared named entities blanked."""
    return ET.fromstring(_NAMED_ENTITY.sub(" ", _DOCTYPE.sub("", text)))


def _text(el):
    return " ".join(" ".join(el.itertext()).split()) if el is not None else ""


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _date(el):
    try:
        y = int(_text(el.find("year")))
    except (ValueError, TypeError):
        return None
    def num(name):
        t = _text(el.find(name))
        return int(t) if t.isdigit() else 1
    try:
        return datetime(y, num("month"), num("day")).strftime("%Y-%m-%d")
    except ValueError:
        try:
            return datetime(y, num("month"), 1).strftime("%Y-%m-%d")
        except ValueError:
            return datetime(y, 1, 1).strftime("%Y-%m-%d")


def parse_nxml(text, fallback_id):
    root = safe_xml(text)
    meta = root.find(".//article-meta")
    if meta is None:
        return None
    ids = {a.get("pub-id-type"): _text(a) for a in meta.findall("article-id")}
    in_file = (ids.get("pmc") or ids.get("pmcid") or "").upper().removeprefix("PMC") or None
    # TREC CDS identifies an article by its file name (its PMCID); the article-id is the fallback.
    by_name = (fallback_id or "").upper().removeprefix("PMC")
    pmc = by_name if by_name.isdigit() else in_file
    if not pmc:
        return None
    abstracts = [a for a in meta.findall("abstract") if a.get("abstract-type") not in ("graphical", "teaser", "toc")]
    dates = {}
    for d in meta.findall("pub-date"):
        kind = d.get("pub-type") or d.get("date-type") or "other"
        dates.setdefault(kind, _date(d))
    pub_date = next((dates[k] for k in ("epub", "ppub") if dates.get(k)), None) or \
        next((v for v in dates.values() if v), None)
    return {"_id": pmc, "pmc_in_file": in_file, "pmid": ids.get("pmid") or None,
            "title": _text(meta.find("title-group/article-title")),
            "abstract": " ".join(_text(a) for a in abstracts).strip(), "pub_date": pub_date,
            "article_type": root.get("article-type")}


def iter_nxml_sources(paths):
    for path in paths:
        if os.path.isdir(path):
            for dirpath, _, files in os.walk(path):
                for fn in sorted(files):
                    if fn.endswith((".nxml", ".xml")):
                        with open(os.path.join(dirpath, fn), "rb") as f:
                            yield fn.rsplit(".", 1)[0], f.read().decode("utf-8", "replace")
        else:
            with tarfile.open(path, "r:*") as tf:
                for m in tf:
                    if m.isfile() and m.name.endswith((".nxml", ".xml")):
                        yield os.path.basename(m.name).rsplit(".", 1)[0], tf.extractfile(m).read().decode("utf-8", "replace")


def parse_topics(text, year):
    root = safe_xml(text)
    out = []
    for t in root.iter("topic"):
        number, kind = t.get("number"), (t.get("type") or "").strip().lower()
        summary = _text(t.find("summary"))
        if not number or not summary:
            raise ValueError(f"{year} topic {number!r}: missing number or summary (A4 requires the summary field)")
        out.append({"qid": f"{year}-{number}", "year": str(year), "number": number, "type": kind,
                    "column": TOPIC_TYPE_TO_COLUMN.get(kind), "summary": summary})
    return out


def parse_qrels(text, year):
    """Four-column TREC qrels. Anything else stops the run (e.g. 2016's five-column sampled file)."""
    q = {}
    for n, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 4 or not re.fullmatch(r"-?\d+", parts[3]):
            raise ValueError(f"{year} qrels line {n}: not a four-column TREC qrel: {line[:80]!r}")
        topic, _, doc, rel = parts
        q.setdefault(f"{year}-{topic}", {})[doc.upper().removeprefix("PMC")] = float(rel)
    return q


def parse_efetch(text):
    root = safe_xml(text)
    out = {}
    for art in root.iter("PubmedArticle"):
        pmid = _text(art.find("MedlineCitation/PMID"))
        if pmid:
            out[pmid] = [_text(p) for p in art.findall(".//PublicationTypeList/PublicationType")]
    return out


# ---------- prepare steps ----------
def write_meta(out_path, extra):
    meta = {"file": os.path.basename(out_path), "sha256": sha256_file(out_path),
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra}
    with open(out_path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def prepare_corpus(sources, out):
    counts, seen = Counter(), set()
    with open(out, "w") as f:
        for fid, text in iter_nxml_sources(sources):
            try:
                doc = parse_nxml(text, fid)
            except ET.ParseError:
                counts["parse_error"] += 1
                continue
            if doc is None:
                counts["no_article_meta"] += 1
                continue
            if doc["_id"] in seen:
                counts["duplicate_id"] += 1
                continue
            seen.add(doc["_id"])
            counts["written"] += 1
            counts["with_abstract"] += bool(doc["abstract"])
            counts["with_date"] += bool(doc["pub_date"])
            counts["with_pmid"] += bool(doc["pmid"])
            counts["file_name_differs_from_article_id"] += bool(doc["pmc_in_file"]) and doc["pmc_in_file"] != doc["_id"]
            f.write(json.dumps(doc) + "\n")
    inputs = [{"path": os.path.basename(s), "sha256": sha256_file(s)} for s in sources if os.path.isfile(s)]
    return write_meta(out, {"step": "prepare-corpus", "counts": dict(counts), "inputs": inputs})


def prepare_pubtypes(corpus, out, batch=200, sleep=None):  # pragma: no cover (network; parse_efetch is tested)
    key = os.environ.get("NCBI_API_KEY", "")
    sleep = sleep if sleep is not None else (0.11 if key else 0.34)
    pmids = sorted({d["pmid"] for d in map(json.loads, open(corpus)) if d.get("pmid")})
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["pmid"] for l in open(out)}
    todo = [p for p in pmids if p not in done]
    log(f"{len(pmids)} PMIDs, {len(done)} already fetched, {len(todo)} to fetch")
    with open(out, "a") as f:
        for i in range(0, len(todo), batch):
            ids = todo[i:i + batch]
            data = urllib.parse.urlencode({"db": "pubmed", "retmode": "xml", "id": ",".join(ids),
                                           **({"api_key": key} if key else {})}).encode()
            for attempt in range(4):
                try:
                    with urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                                                data=data, timeout=120) as r:
                        got = parse_efetch(r.read().decode("utf-8", "replace"))
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        raise
                    time.sleep(2 ** (attempt + 1))
            for p in ids:
                f.write(json.dumps({"pmid": p, "types": got.get(p)}) + "\n")
            f.flush()
            time.sleep(sleep)
            if (i // batch) % 50 == 0:
                log(f"  {i + len(ids)}/{len(todo)}")
    return write_meta(out, {"step": "prepare-pubtypes", "pmids": len(pmids)})


# ---------- A6: sparse BM25 with the repository's formula, tokenizer and tie order ----------
class SparseBM25:
    TOKEN = re.compile(r"\b\w+\b")

    def __init__(self, texts, k1=1.5, b=0.75):
        """texts: any iterable, read once. Term counts are held as 32-bit integers (about two
        million abstracts fit in memory) and cast to float64 when scored, as the repository does."""
        self.k1, self.b = k1, b
        vocab, rows, cols, vals, lengths = {}, array.array("i"), array.array("i"), array.array("i"), array.array("q")
        for i, text in enumerate(texts):
            toks = self.TOKEN.findall(text.lower())
            lengths.append(len(toks))
            for term, n in Counter(toks).items():
                rows.append(vocab.setdefault(term, len(vocab))); cols.append(i); vals.append(n)
        self.vocab, self.N = vocab, len(lengths)
        self.avgdl = sum(lengths) / max(len(lengths), 1)
        self.dl = np.asarray(lengths, dtype=np.float64)
        self.tf = sparse.csr_matrix((np.frombuffer(vals, dtype=np.int32), (np.frombuffer(rows, dtype=np.int32),
                                     np.frombuffer(cols, dtype=np.int32))), shape=(len(vocab), self.N))
        self.tf.sort_indices()
        del rows, cols, vals

    def scores(self, query):
        s = np.zeros(self.N)
        for term in self.TOKEN.findall(query.lower()):
            j = self.vocab.get(term)
            if j is None:
                continue
            a, z_ = self.tf.indptr[j], self.tf.indptr[j + 1]
            docs, tf = self.tf.indices[a:z_], self.tf.data[a:z_].astype(np.float64)
            df = int(z_ - a)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * self.dl[docs] / self.avgdl))
            s[docs] += idf * tf_norm
        return s

    def search(self, query, k):
        s = self.scores(query)
        return [int(i) for i in np.argsort(-s, kind="stable")[:k]], s


def a6_gate(nf_dir, expected):
    corpus = [json.loads(l) for l in open(os.path.join(nf_dir, "corpus.jsonl"))]
    qtext = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(os.path.join(nf_dir, "queries.jsonl"))}
    qrels = {}
    with open(os.path.join(nf_dir, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            q, d, g = line.rstrip("\n").split("\t")
            qrels.setdefault(q, {})[d] = float(g)
    ref = datetime(2020, 1, 1)
    docs = [Document(id=r["_id"], title=r["title"], abstract=r["text"], content_embedding=np.zeros(1),
                     publication_date=ref) for r in corpus]
    repo = BM25(); repo.index(docs)
    sp = SparseBM25([f"{r['title']} {r['text']}" for r in corpus])
    ids = [r["_id"] for r in corpus]
    mism, nd = 0, []
    for q in sorted(q for q in qrels if q in qtext):
        a = [i for i, _ in repo.search(qtext[q], POOL)]
        b, _ = sp.search(qtext[q], POOL)
        mism += a != b
        nd.append(ndcg_at_k([ids[i] for i in b], qrels[q], K))
    g = {"queries": len(nd), "top100_mismatches": mism, "ndcg10": float(np.mean(nd)), "expected": expected,
         "sha256": {f: sha256_file(os.path.join(nf_dir, f)) for f in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")}}
    g["passed"] = mism == 0 and (expected is None or abs(g["ndcg10"] - expected) < 1e-9)
    return g


# ---------- encoders ----------
class Encoder:
    def __init__(self, cache, fake=False):
        self.cache, self.fake, self.paths = cache, fake, {}
        os.makedirs(cache, exist_ok=True)

    def __call__(self, inputs, key):
        name, rev, pooling, max_len = ENCODERS[key]
        h = hashlib.sha256(f"{name}|{rev}|{pooling}|{max_len}|fake={self.fake}\n".encode())
        for t in inputs:
            h.update((json.dumps(t) + "\n").encode())
        path = os.path.join(self.cache, f"emb_{key}_{h.hexdigest()[:16]}.npy")
        self.paths[key + ":" + h.hexdigest()[:16]] = path
        if os.path.exists(path):
            return np.load(path)
        if self.fake:  # stand-in for check_stage1.py only: deterministic vectors from the text
            arr = np.vstack([np.random.default_rng(int(hashlib.sha256(json.dumps(t).encode()).hexdigest()[:8], 16))
                             .standard_normal(16) for t in inputs]).astype(np.float32)
        else:
            arr = self._model(inputs, name, rev, pooling, max_len)
        np.save(path, arr)
        return arr

    @staticmethod
    def _model(inputs, name, rev, pooling, max_len):  # pragma: no cover (needs the models)
        import torch
        from transformers import AutoTokenizer, AutoModel
        torch.manual_seed(0)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(name, revision=rev)
        mdl = AutoModel.from_pretrained(name, revision=rev).eval().to(dev)
        out = []
        for i in range(0, len(inputs), 32):
            enc = tok(inputs[i:i + 32], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
            with torch.no_grad():
                hs = mdl(**enc).last_hidden_state
            if pooling == "cls":
                v = hs[:, 0, :]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(v.float().cpu().numpy())
        return np.vstack(out)


# ---------- scoring helpers ----------
def z(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return np.zeros_like(x) if sd == 0 else (x - x.mean()) / sd


def fill_pool_mean(x):
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return np.zeros_like(x)
    return np.where(np.isnan(x), np.nanmean(x), x)


def gated(Z, gates, off):
    return sum((0 if m in off else gates[m]) * Z[i] for i, m in enumerate(MODULES))


def rank_by(score, pool_ids):
    return [pool_ids[i] for i in np.argsort(-np.asarray(score), kind="stable")]


def l_features(Z, column):
    ind = np.array([1.0 if t == column else 0.0 for t in TYPES])
    return np.hstack([Z.T, np.kron(ind, Z.T)])


def fit_rank(X, lam):
    def f(w):
        m = X @ w
        return (np.mean(np.logaddexp(0, -m)) + lam * w @ w,
                -(X.T @ (1 / (1 + np.exp(m)))) / len(m) + 2 * lam * w)
    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def pairs_for(items, rng):
    X = []
    for feats, cand in items:
        if len(cand) > MAX_PAIRS:
            cand = cand[rng.choice(len(cand), MAX_PAIRS, replace=False)]
        if len(cand):
            X.append(feats[cand[:, 0]] - feats[cand[:, 1]])
    return np.vstack(X) if X else np.zeros((0, 30))


def cv_choose_lambda(items, ndcg_of):
    """items: list of (feats, cand, key); ndcg_of(key, scores) -> NDCG. 5-fold CV, seed 0."""
    folds = np.array_split(np.random.default_rng(0).permutation(len(items)), 5)
    cv = {}
    for lam in LAMBDAS:
        vals = []
        for k in range(5):
            held = set(folds[k].tolist())
            Xp = pairs_for([(f, c) for j, (f, c, _) in enumerate(items) if j not in held], np.random.default_rng(0))
            if not len(Xp):
                continue
            w = fit_rank(Xp, lam)
            vals += [ndcg_of(items[j][2], items[j][0] @ w) for j in held]
        cv[lam] = float(np.mean(vals)) if vals else float("-inf")
    best = max(LAMBDAS, key=lambda l: (cv[l], -LAMBDAS.index(l)))
    return best, cv


# ---------- statistics and the decision rule ----------
def paired(diffs, seed=0):
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    null = np.abs((rng.choice([-1.0, 1.0], size=(N_PERM, len(d))) * d).mean(1))
    p = (1 + np.sum(null >= abs(d.mean()) - 1e-15)) / (N_PERM + 1)
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)], "p": float(p), "n": int(len(d)),
            "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "favour": [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())]}


def holm(pvals):
    order, adj, running = np.argsort(pvals), np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i]); adj[i] = min(1.0, running)
    return adj.tolist()


def decide(primary, sec, controls, manipulation_ok):
    """The registered decision rule with Amendment 1 (A8) and Amendment 2 (B2), in order."""
    sig = lambda k: sec[k]["p_holm"] < 0.05 and sec[k]["mean"] > 0
    gain = primary["mean"] >= 0.05 and primary["ci95"][0] > 0
    n_ok = controls["Nbar-B0"]["ci95"][1] < 0.05
    if controls["Pbar-M"]["ci95"][0] <= 0:
        return "Test failed"
    if primary["ci95"][1] < 0.05:
        return "Refuted"
    if not manipulation_ok:
        return "Untestable"
    if gain and sig("M-F") and sig("M-U") and sig("M-V") and n_ok:
        return "Supported"
    if gain and (not sig("M-U") or not sig("M-V") or not n_ok):
        return "Gain without attribution"
    return "Inconclusive"


def analyze(A, manipulation_ok):
    """A: per-topic NDCG arrays for B0, F, M, U, V, L, N0..N4, R0..R4, P0..P4 (same topic order)."""
    Nbar = np.mean([A[f"N{s}"] for s in SEEDS], 0)
    Rbar = np.mean([A[f"R{s}"] for s in SEEDS], 0)
    Pbar = np.mean([A[f"P{s}"] for s in SEEDS], 0)
    primary = paired(A["M"] - A["B0"])
    fam = {"M-F": A["M"] - A["F"], "M-U": A["M"] - A["U"], "M-V": A["M"] - A["V"],
           "M-L": A["M"] - A["L"], "M-Rbar": A["M"] - Rbar}
    sec = {k: paired(v) for k, v in fam.items()}
    for k, p in zip(sec, holm([sec[k]["p"] for k in sec])):
        sec[k]["p_holm"] = p
    controls = {"Nbar-B0": paired(Nbar - A["B0"]), "Pbar-M": paired(Pbar - A["M"])}
    verdict = decide(primary, sec, controls, manipulation_ok)
    qualifiers = []
    if sec["M-L"]["p_holm"] < 0.05 and sec["M-L"]["mean"] < 0:
        qualifiers.append("A learned combination (L) beats the designed one (M).")
    if not (sec["M-Rbar"]["p_holm"] < 0.05 and sec["M-Rbar"]["mean"] > 0):
        qualifiers.append("The specific question-type mapping adds nothing detectable beyond gating as such (M - Rbar).")
    means = {k: float(np.mean(v)) for k, v in A.items()}
    means.update({"N_bar": float(Nbar.mean()), "R_bar": float(Rbar.mean()), "P_bar": float(Pbar.mean())})
    return {"means": means, "primary_M_minus_B0": primary, "secondary_holm": sec, "controls": controls,
            "verdict": verdict, "qualifiers": qualifiers}


def manipulation_check(topics, top10, evidence_varies):
    diag = [t["qid"] for t in topics if t["column"] == "diagnosis_test"]
    differ = [top10["M"][q] != top10["U"][q] for q in diag]
    share = float(np.mean(differ)) if diag else 0.0
    return {"diagnosis_test_topics": len(diag), "share_M_top10_differs_from_U": share,
            "share_evidence_varies_in_pool": float(np.mean([evidence_varies[q] for q in diag])) if diag else 0.0,
            "threshold": MANIPULATION_MIN_SHARE, "passed": bool(diag) and share >= MANIPULATION_MIN_SHARE}


# ---------- the run ----------
def resolve(cfg_dir, p):
    return p if os.path.isabs(p) else os.path.join(cfg_dir, p)


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def versions():
    v = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    for mod in ("torch", "transformers"):
        try:
            v[mod] = __import__(mod).__version__
        except ImportError:
            v[mod] = None
    return v


def run(config_path, out, cache, test_mode=False, fake_encoder=False):
    if fake_encoder and not test_mode:
        raise SystemExit("The stand-in encoder is for check_stage1.py only (Amendment 2, B4).")
    t0, started, events = time.time(), datetime.now(timezone.utc).isoformat(timespec="seconds"), []
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    cfg = json.load(open(config_path))
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    R = {"stage": "1", "test_mode": test_mode, "fidelity": {}, "preconditions": {}}

    # 1. Gates (A5 is in check_stage1.py and repeated here; A6 on NFCorpus test).
    bad = [c for c, e in [(["Randomized Controlled Trial", "Clinical Trial"], EvidenceLevel.RCT),
                          (["Meta-Analysis"], EvidenceLevel.SYSTEMATIC_REVIEW), (["Review"], None)] if pt_to_level(c) != e]
    assert not bad, f"A5 mapping failed on {bad}"
    gate = a6_gate(resolve(cfg_dir, cfg["nfcorpus"]), None if test_mode else cfg.get("nfcorpus_expected_ndcg10", AUGUST_BM25_NDCG))
    R["fidelity"]["a6_sparse_bm25_gate"] = gate
    log(f"A6 gate: {gate}", events)
    if not gate["passed"]:
        raise SystemExit("A6 sparse BM25 fidelity gate failed; Stage 1 stops here")

    # 2. Data checksums (B6) and reference dates (must be given).
    collections, data_sha = {}, {}
    for name, c in cfg["collections"].items():
        if not c.get("reference_date"):
            raise SystemExit(f"{name}: reference_date must be set in the configuration (B6)")
        for key in ("corpus", "pubtypes"):
            p = resolve(cfg_dir, c[key])
            meta = json.load(open(p + ".meta.json"))
            actual = sha256_file(p)
            if actual != meta["sha256"]:
                raise SystemExit(f"{p}: SHA-256 {actual} does not match its .meta.json ({meta['sha256']})")
            data_sha[f"{name}/{key}"] = actual
        collections[name] = c
    data_sha.update({f"nfcorpus/{k}": v for k, v in gate["sha256"].items()})
    log("Data checksums verified", events)

    # 3. Topics (no judgments).
    topics = []
    for year, y in sorted(cfg["years"].items()):
        if y["collection"] not in collections:
            raise SystemExit(f"{year}: collection {y['collection']!r} is not defined in the configuration")
        p = resolve(cfg_dir, y["topics"]); data_sha[f"{year}/topics"] = sha256_file(p)
        for t in parse_topics(open(p, encoding="utf-8").read(), year):
            t["collection"] = y["collection"]; topics.append(t)
    R["topics"] = {"n": len(topics), "by_type": dict(Counter(t["type"] for t in topics)),
                   "by_column": dict(Counter(str(t["column"]) for t in topics))}
    unmapped = [t["qid"] for t in topics if t["column"] is None]
    if unmapped:
        raise SystemExit(f"Topics with a type outside diagnosis/test/treatment: {unmapped[:5]}")
    log(f"Topics: {R['topics']}", events)

    # 4. Collections, pools and module inputs.
    enc = Encoder(os.path.join(cache), fake=fake_encoder)
    pools, raw, Zs, off, ref_dates, docs_of = {}, {}, {}, {}, {}, {}
    temporal_on, cov = {}, {}
    for name, c in collections.items():
        ts = [t for t in topics if t["collection"] == name]
        if not ts:
            continue
        cpath = resolve(cfg_dir, c["corpus"])
        dated = []  # one pass builds the index; only pooled documents are kept in memory
        def texts():
            for d in iter_jsonl(cpath):
                dated.append(bool(d.get("pub_date")))
                yield f"{d['title']} {d['abstract']}"
        bm = SparseBM25(texts())
        ref = datetime.strptime(c["reference_date"], "%Y-%m-%d"); ref_dates[name] = c["reference_date"]
        cov[name] = float(np.mean(dated))
        temporal_on[name] = cov[name] >= DATE_COVERAGE_MIN
        log(f"{name}: {bm.N} documents, date coverage {cov[name]:.4f}, temporal "
            f"{'on' if temporal_on[name] else 'off (criterion 1)'}", events)
        pool_idx, scores_of = {}, {}
        for t in ts:
            pool_idx[t["qid"]], s = bm.search(t["summary"], POOL)
            scores_of[t["qid"]] = s[pool_idx[t["qid"]]]
        del bm
        needed = {i for p in pool_idx.values() for i in p}
        row = {i: d for i, d in enumerate(iter_jsonl(cpath)) if i in needed}
        pubtypes = {r["pmid"]: r["types"] for r in iter_jsonl(resolve(cfg_dir, c["pubtypes"]))}
        for t in ts:
            q = t["qid"]
            docs_of[q] = [row[i] for i in pool_idx[q]]
            pools[q] = [d["_id"] for d in docs_of[q]]
            raw[q] = {"bm25": scores_of[q], "ref": ref, "pubtypes": [pubtypes.get(d.get("pmid")) for d in docs_of[q]]}
            off[q] = {"applicability"} | (set() if temporal_on[name] else {"temporal"})
    R["preconditions"]["date_coverage"] = {n: {"coverage": cov[n], "temporal_on": temporal_on[n]} for n in cov}

    # 5. Embeddings for pooled documents and topic summaries only.
    uniq = {}
    for q in pools:
        for d in docs_of[q]:
            uniq.setdefault(d["_id"], d)
    uid = list(uniq)
    E_art = dict(zip(uid, enc([[uniq[i]["title"], uniq[i]["abstract"]] for i in uid], "medcpt_article")))
    E_pmb = dict(zip(uid, enc([f"{uniq[i]['title']} {uniq[i]['abstract']}".strip() for i in uid], "pubmedbert")))
    qids = [t["qid"] for t in topics]
    E_q = dict(zip(qids, enc([t["summary"] for t in topics], "medcpt_query")))
    E_qp = dict(zip(qids, enc([t["summary"] for t in topics], "pubmedbert")))
    log(f"Embedded {len(uid)} pooled documents and {len(qids)} topics", events)

    # 6. Module scores.
    labelled, evidence_varies = [], {}
    for t in topics:
        q, r, ds = t["qid"], raw[t["qid"]], docs_of[t["qid"]]
        levels = [pt_to_level(p) for p in r["pubtypes"]]
        labelled += [lv is not None for lv in levels]
        ev = fill_pool_mean([evidence_weight(lv) if lv is not None else np.nan for lv in levels])
        evidence_varies[q] = bool(np.ptp(ev) > 0)
        tau = fill_pool_mean([temporal_decay(datetime.strptime(d["pub_date"], "%Y-%m-%d"), r["ref"], "pbm")
                              if d.get("pub_date") else np.nan for d in ds])
        raw_mod = {"bm25": r["bm25"], "topical": np.array([cosine_similarity(E_art[d["_id"]], E_q[q]) for d in ds]),
                   "evidence": ev, "temporal": tau, "applicability": np.zeros(len(ds)),
                   "actionability": np.array([actionability_relevance(Document(id=d["_id"], title=d["title"],
                                              abstract=d["abstract"], content_embedding=np.zeros(1),
                                              publication_date=r["ref"], evidence_level=EvidenceLevel.RCT), None) for d in ds])}
        raw[q]["modules"] = raw_mod
        Zs[q] = np.vstack([z(raw_mod[m]) for m in MODULES])

    # 7. Judgment-free rankings (B3).
    perm = {s: np.random.default_rng(s).permutation(len(TYPES)) for s in SEEDS}
    R["R_permutations"] = {str(s): [TYPES[i] for i in perm[s]] for s in SEEDS}
    rankings = {a: {} for a in ["B0", "F", "M", "U", "V"] + [f"N{s}" for s in SEEDS] + [f"R{s}" for s in SEEDS]}
    checks = Counter()
    only_bm25 = {m: int(m == "bm25") for m in MODULES}
    for qi, t in enumerate(topics):
        q, col, Z, pids, ds = t["qid"], t["column"], Zs[t["qid"]], pools[t["qid"]], docs_of[t["qid"]]
        rankings["B0"][q] = pids
        checks["M_bm25_only_equals_B0"] += rank_by(gated(Z, only_bm25, off[q]), pids) == pids
        rankings["M"][q] = rank_by(gated(Z, GATES[col], off[q]), pids)
        rankings["U"][q] = rank_by(gated(Z, {m: 1 for m in MODULES}, off[q]), pids)
        rankings["V"][q] = rank_by(gated(Z, {m: int(m != "evidence") for m in MODULES}, off[q]), pids)
        qF = Query(text=t["summary"], embedding=E_qp[q])
        ref = raw[q]["ref"]
        dF = [Document(id=d["_id"], title=d["title"], abstract=d["abstract"], content_embedding=E_pmb[d["_id"]],
                       publication_date=datetime.strptime(d["pub_date"], "%Y-%m-%d") if d.get("pub_date") else ref)
              for d in ds]
        sF = [clinical_relevance(d, qF, query_time=ref) for d in dF]
        indep = [temporal_decay(d.publication_date, ref, "pbm") * evidence_weight(d.evidence_level)
                 * (0.5 * cosine_similarity(d.content_embedding, qF.embedding) + 0.3 * 0.5
                    + 0.2 * actionability_relevance(d, qF)) for d in dF]
        checks["F_formula_mismatch"] += int(max(abs(a - b) for a, b in zip(sF, indep)) > 1e-12)
        rankings["F"][q] = [d.id for d, _ in sorted(zip(dF, sF), key=lambda x: x[1], reverse=True)]
        for s in SEEDS:
            nrng = np.random.default_rng([s, qi, 11])
            ZN = np.vstack([Z[0]] + [z(nrng.standard_normal(len(pids))) for _ in MODULES[1:]])
            rankings[f"N{s}"][q] = rank_by(gated(ZN, GATES[col], off[q]), pids)
            rankings[f"R{s}"][q] = rank_by(gated(Z, GATES[TYPES[perm[s][TYPES.index(col)]]], off[q]), pids)
        if col == "treatment":
            checks["M_equals_U_on_treatment"] += rankings["M"][q] == rankings["U"][q]
    n_treat = sum(t["column"] == "treatment" for t in topics)
    R["fidelity"]["checks"] = {"M_bm25_only_equals_B0": f"{checks['M_bm25_only_equals_B0']}/{len(topics)}",
                               "M_equals_U_on_treatment": f"{checks['M_equals_U_on_treatment']}/{n_treat}",
                               "F_formula_mismatch": checks["F_formula_mismatch"]}
    log(f"Fidelity checks: {R['fidelity']['checks']}", events)
    if (checks["M_bm25_only_equals_B0"] != len(topics) or checks["M_equals_U_on_treatment"] != n_treat
            or checks["F_formula_mismatch"]):
        raise SystemExit("A fidelity check failed; Stage 1 stops before any judgment is read")
    for arm, rk in rankings.items():
        with open(os.path.join(out, "runs", f"{arm}.trec"), "w") as f:
            for q in qids:
                for r_, d in enumerate(rk[q]):
                    f.write(f"{q} Q0 {d} {r_ + 1} {POOL - r_} {arm}\n")
    log("Judgment-free rankings saved", events)

    # 8. Manipulation check (B2), saved before any judgment is read.
    manip = manipulation_check(topics, {a: {q: rankings[a][q][:K] for q in qids} for a in ("M", "U")}, evidence_varies)
    manip["share_pooled_documents_with_evidence_label"] = float(np.mean(labelled)) if labelled else 0.0
    with open(os.path.join(out, "manipulation_check.json"), "w") as f:
        json.dump(manip, f, indent=2)
    R["preconditions"]["manipulation_check"] = manip
    log(f"Manipulation check saved: {manip}", events)

    # 9. Judgments, read only now.
    qrels = {}
    for year, y in sorted(cfg["years"].items()):
        p = resolve(cfg_dir, y["qrels"]); data_sha[f"{year}/qrels"] = sha256_file(p)
        qrels.update(parse_qrels(open(p, encoding="utf-8").read(), year))
    log("Judgments read", events)
    missing = [q for q in qids if q not in qrels]
    if missing:
        raise SystemExit(f"Topics without judgments: {missing[:5]}")
    levels_present = sorted({g for q in qids for g in qrels[q].values()})
    noise_s = (max(levels_present) - min(levels_present)) / 2 if len(levels_present) > 1 else 1.0  # B5

    # 10. L, cross-fitted over topics (A12/B5), and P.
    def grades_of(q):
        return np.array([qrels[q].get(d, 0.0) for d in pools[q]])
    items = {q: (l_features(Zs[q], t["column"]), np.argwhere(grades_of(q)[:, None] > grades_of(q)[None, :]))
             for t in topics for q in [t["qid"]]}
    nd_scores = lambda q, sc: ndcg_at_k(rank_by(sc, pools[q]), qrels[q], K)
    outer = np.array_split(np.random.default_rng(0).permutation(len(qids)), 5)
    rankings["L"], L_info = {}, []
    for k in range(5):
        test_q = [qids[j] for j in outer[k]]
        train_q = [q for q in qids if q not in set(test_q)]
        lam, cv = cv_choose_lambda([(items[q][0], items[q][1], q) for q in train_q], nd_scores)
        Xp = pairs_for([items[q] for q in train_q], np.random.default_rng(0))
        w = fit_rank(Xp, lam) if len(Xp) else np.zeros(30)
        for q in test_q:
            rankings["L"][q] = rank_by(items[q][0] @ w, pools[q])
        L_info.append({"fold": k, "lambda": lam, "cv": {str(a): b for a, b in cv.items()},
                       "at_lower_boundary": lam == min(LAMBDAS)})
    R["L"] = {"folds": L_info, "boundary_limitation": any(f["at_lower_boundary"] for f in L_info)}
    for s in SEEDS:
        rankings[f"P{s}"] = {}
        for qi, t in enumerate(topics):
            q = t["qid"]
            prng = np.random.default_rng([s, qi, 13])
            synth = z(grades_of(q) + noise_s * prng.standard_normal(len(pools[q])))
            rankings[f"P{s}"][q] = rank_by(gated(Zs[q], GATES[t["column"]], off[q]) + synth, pools[q])
    for arm in ["L"] + [f"P{s}" for s in SEEDS]:
        with open(os.path.join(out, "runs", f"{arm}.trec"), "w") as f:
            for q in qids:
                for r_, d in enumerate(rankings[arm][q]):
                    f.write(f"{q} Q0 {d} {r_ + 1} {POOL - r_} {arm}\n")

    # 11. Scores, statistics, verdict.
    A = {arm: np.array([ndcg_at_k(rankings[arm][q], qrels[q], K) for q in qids]) for arm in rankings}
    R.update(analyze(A, manip["passed"]))
    R["p_noise_sd"] = noise_s
    R["runtime_seconds"] = round(time.time() - t0, 1)
    R["events"] = events
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(R, f, indent=2)
    names = list(rankings)
    with open(os.path.join(out, "per_query.csv"), "w", newline="") as f:
        w_ = csv.writer(f); w_.writerow(["qid", "type", "column"] + names)
        for i, t in enumerate(topics):
            w_.writerow([t["qid"], t["type"], t["column"]] + [f"{A[n][i]:.10f}" for n in names])
    write_report(R, os.path.join(out, "RESULTS.md"))
    try:
        commit = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        commit = None
    manifest = {"stage": "1", "test_mode": test_mode, "started": started,
                "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code_commit": commit,
                "script_sha256": sha256_file(os.path.abspath(__file__)), "config": cfg, "data_sha256": data_sha,
                "encoders": {k: {"name": v[0], "revision": v[1], "pooling": v[2], "max_length": v[3]} for k, v in ENCODERS.items()},
                "fake_encoder": fake_encoder,
                "embeddings_sha256": {os.path.basename(p): sha256_file(p) for p in enc.paths.values() if os.path.exists(p)},
                "environment": {"platform": platform.platform(), "cpus": os.cpu_count(), **versions()}}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"Verdict (Stage 1{', TEST MODE' if test_mode else ''}): {R['verdict']}", events)
    return R


def write_report(R, path):
    """Generated report: every number comes from R."""
    f4 = lambda x: f"{x:+.4f}"
    ci = lambda d: f"[{f4(d['ci95'][0])}, {f4(d['ci95'][1])}]"
    m = R["preconditions"]["manipulation_check"]
    lines = [f"# Stage 1 results{' (TEST MODE, not a result)' if R['test_mode'] else ''}", "",
             "Generated by `stage1.py` from `results.json`. Do not edit by hand.", "",
             f"**Verdict under the registered decision rule: {R['verdict']}.**", ""]
    lines += [f"- {q}" for q in R["qualifiers"]]
    if R["L"]["boundary_limitation"]:
        lines.append("- L chose the smallest penalty in the grid in at least one fold; reported as a limitation (B1).")
    lines += ["", "## Preconditions", "",
              f"- Manipulation check (B2): M's top 10 differs from U's on {m['share_M_top10_differs_from_U']:.3f} of "
              f"{m['diagnosis_test_topics']} diagnosis/test topics (threshold {m['threshold']}): "
              f"{'passed' if m['passed'] else 'FAILED'}.",
              f"- Evidence varies within the pool on {m['share_evidence_varies_in_pool']:.3f} of those topics; "
              f"{m['share_pooled_documents_with_evidence_label']:.3f} of pooled documents carry an evidence label."]
    for n, c in R["preconditions"]["date_coverage"].items():
        lines.append(f"- {n}: date coverage {c['coverage']:.4f}; temporal {'on' if c['temporal_on'] else 'off (criterion 1)'}.")
    lines += ["", "## Comparisons", "", "| Comparison | Mean | 95% CI | p (Holm) |", "|---|---|---|---|",
              f"| **M - B0 (primary)** | {f4(R['primary_M_minus_B0']['mean'])} | {ci(R['primary_M_minus_B0'])} | "
              f"{R['primary_M_minus_B0']['p']:.4f} (uncorrected) |"]
    lines += [f"| {k} | {f4(v['mean'])} | {ci(v)} | {v['p_holm']:.4f} |" for k, v in R["secondary_holm"].items()]
    lines += [f"| {k} (control) | {f4(v['mean'])} | {ci(v)} | {v['p']:.4f} (uncorrected) |" for k, v in R["controls"].items()]
    lines += ["", "## Means (NDCG@10)", "", "| Arm | NDCG@10 |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" for k, v in R["means"].items()]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("prepare-corpus"); pc.add_argument("sources", nargs="+"); pc.add_argument("--out", required=True)
    pp = sub.add_parser("prepare-pubtypes"); pp.add_argument("--corpus", required=True); pp.add_argument("--out", required=True)
    pp.add_argument("--batch", type=int, default=200)
    rr = sub.add_parser("run"); rr.add_argument("--config", required=True); rr.add_argument("--out", required=True)
    rr.add_argument("--cache", required=True); rr.add_argument("--test-mode", action="store_true")
    rr.add_argument("--fake-encoder", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prepare-corpus":
        print(json.dumps(prepare_corpus(a.sources, a.out), indent=2))
    elif a.cmd == "prepare-pubtypes":
        print(json.dumps(prepare_pubtypes(a.corpus, a.out, a.batch), indent=2))
    else:
        run(a.config, a.out, a.cache, a.test_mode, a.fake_encoder)


if __name__ == "__main__":
    main()

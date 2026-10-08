"""
Stage 0 pilot (exploratory) for ../PREREGISTRATION.md, as fixed by Amendment 1 (A6, A7, A12).
It checks the plumbing and the controls. It is never evidence for the claim.

    HF_HUB_DISABLE_XET=1 python evals/modulated-composition-2026-10-04/stage0/stage0.py \
        --data <dir with corpus.jsonl, queries.jsonl, qrels/{train,dev,test}.tsv> \
        --cache <embedding cache dir> [--pubmedbert-doc-cache <audit emb_docs_*.npy>] \
        --repro-bm25 0.3103001666796228

Order of work: fidelity gates first (A6 sparse BM25, A5 mapping unit tests); if any fails,
the script stops before any dev-query result is computed.
"""
import argparse, csv, hashlib, json, math, os, platform, re, sys, time
from collections import Counter
from datetime import datetime, timezone
import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from clinical_ir.relevance import Document, Query, clinical_relevance, cosine_similarity, actionability_relevance
from clinical_ir.retrieval import BM25
from clinical_ir.evaluate import ndcg_at_k
from clinical_ir.evidence import EvidenceLevel, evidence_weight

HERE = os.path.dirname(os.path.abspath(__file__))
K, POOL = 10, 100
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = N_PERM = 10_000
LAMBDAS = [0.01, 0.1, 1, 10]
MAX_PAIRS = 500

MODULES = ["bm25", "topical", "evidence", "temporal", "applicability", "actionability"]
# Registered gate table (PREREGISTRATION.md, "Gate table").
GATES = {
    "treatment":      dict(bm25=1, topical=1, evidence=1, temporal=1, applicability=1, actionability=1),
    "diagnosis_test": dict(bm25=1, topical=1, evidence=0, temporal=1, applicability=1, actionability=1),
    "prognosis":      dict(bm25=1, topical=1, evidence=0, temporal=1, applicability=1, actionability=0),
    "other":          dict(bm25=1, topical=1, evidence=0, temporal=0, applicability=0, actionability=0),
}
TYPES = list(GATES)
GLOBALLY_OFF = {"applicability"}  # A12: applicability is off on NFCorpus (no population metadata)
# Registered question-type rule, applied to lower-cased text, in this order.
RULES = [("treatment", r"\b(treat|therap|drug|dose|dosing|supplement|intervention|prevent)"),
         ("diagnosis_test", r"\b(diagnos|tests?\b|testing\b|screen|detect|marker|imaging)"),
         ("prognosis", r"\b(prognos|survival|risk of|outcome|mortality|recurrence)")]

ENCODERS = {
    "medcpt_query": ("ncbi/MedCPT-Query-Encoder", "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc", "cls", 64),
    "medcpt_article": ("ncbi/MedCPT-Article-Encoder", "d05a736da4bb84ee4057b7f7999485be6ed85465", "cls", 512),
    "pubmedbert": ("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                   "e1354b7a3a09615f6aba48dfad4b7a613eef7062", "mean", 512),
}
AUDIT_PUBMEDBERT_DOCS_SHA = "423d11c9c3baf619b4ecdea02665dc9d7b9444a4964553ccca065a14506d1a9a"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def question_type(text):
    t = text.lower()
    for name, pattern in RULES:
        if re.search(pattern, t):
            return name
    return "other"


# ---------- A5: MEDLINE publication type -> evidence level (unit-tested here; unused on NFCorpus) ----------
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
    levels = [PT_LEVEL[p] for p in pub_types if p in PT_LEVEL]
    return min(levels) if levels else None  # IntEnum: lower value = higher level


def mapping_unit_tests():
    cases = [
        (["Randomized Controlled Trial", "Clinical Trial", "Comparative Study", "Journal Article"], EvidenceLevel.RCT),
        (["Meta-Analysis", "Review"], EvidenceLevel.SYSTEMATIC_REVIEW),
        (["Systematic Review", "Randomized Controlled Trial"], EvidenceLevel.SYSTEMATIC_REVIEW),
        (["Clinical Trial"], EvidenceLevel.CONTROLLED_OBSERVATIONAL),
        (["Observational Study", "Journal Article"], EvidenceLevel.CONTROLLED_OBSERVATIONAL),
        (["Case Reports"], EvidenceLevel.UNCONTROLLED_OBSERVATIONAL),
        (["Editorial", "Comment"], EvidenceLevel.EXPERT_OPINION),
        (["Review"], None), (["Journal Article"], None), ([], None),
    ]
    return [{"input": c, "expected": getattr(e, "name", None), "got": getattr(pt_to_level(c), "name", None),
             "ok": pt_to_level(c) == e} for c, e in cases]


# ---------- A6: sparse BM25 with the repository's formula, tokenizer and tie order ----------
class SparseBM25:
    TOKEN = re.compile(r"\b\w+\b")

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        vocab, rows, cols, vals, lengths = {}, [], [], [], []
        for i, text in enumerate(texts):
            toks = self.TOKEN.findall(text.lower())
            lengths.append(len(toks))
            for term, n in Counter(toks).items():
                rows.append(vocab.setdefault(term, len(vocab))); cols.append(i); vals.append(n)
        self.vocab, self.N = vocab, len(texts)
        self.avgdl = sum(lengths) / max(len(lengths), 1)
        self.dl = np.asarray(lengths, dtype=np.float64)
        self.tf = sparse.csr_matrix((np.asarray(vals, dtype=np.float64), (rows, cols)), shape=(len(vocab), self.N))
        self.tf.sort_indices()

    def scores(self, query):
        s = np.zeros(self.N)
        for term in self.TOKEN.findall(query.lower()):  # duplicates count, as in the repository
            j = self.vocab.get(term)
            if j is None:
                continue
            a, z = self.tf.indptr[j], self.tf.indptr[j + 1]
            docs, tf = self.tf.indices[a:z], self.tf.data[a:z]
            df = int(z - a)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * self.dl[docs] / self.avgdl))
            s[docs] += idf * tf_norm
        return s

    def search(self, query, k):
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [int(i) for i in order], s


# ---------- encoders ----------
def embed(texts, key, cache_dir, pairs=False):
    name, rev, pooling, max_len = ENCODERS[key]
    h = hashlib.sha256(f"{name}|{rev}|{pooling}|{max_len}|{pairs}\n".encode())
    for t in texts:
        h.update((json.dumps(t) + "\n").encode())
    path = os.path.join(cache_dir, f"emb_{key}_{h.hexdigest()[:16]}.npy")
    if os.path.exists(path):
        return np.load(path), path
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    mdl = AutoModel.from_pretrained(name, revision=rev).eval()
    out, bs = [], 32
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            hs = mdl(**enc).last_hidden_state
        if pooling == "cls":
            v = hs[:, 0, :]
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(v.numpy().astype(np.float32))
        if (i // bs) % 25 == 0:
            log(f"  [{key}] {i + len(batch)}/{len(texts)}")
    arr = np.vstack(out)
    np.save(path, arr)
    return arr, path


# ---------- statistics (as in the audit) ----------
def paired(diffs, seed=0):
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    null = np.abs((rng.choice([-1.0, 1.0], size=(N_PERM, len(d))) * d).mean(1))
    p = (1 + np.sum(null >= abs(d.mean()) - 1e-15)) / (N_PERM + 1)
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)], "p": float(p), "n": int(len(d)),
            "sd": float(d.std(ddof=1)), "favour": [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())]}


def holm(pvals):
    order, adj, running = np.argsort(pvals), np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i]); adj[i] = min(1.0, running)
    return adj.tolist()


def z(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return np.zeros_like(x) if sd == 0 else (x - x.mean()) / sd


def rank_by(score, pool_ids):
    order = np.argsort(-np.asarray(score), kind="stable")  # ties keep the pool's BM25 order
    return [pool_ids[i] for i in order]


# ---------- pairwise logistic ranker (L) ----------
def l_features(Z, qtype):
    ind = np.array([1.0 if t == qtype else 0.0 for t in TYPES])
    return np.hstack([Z.T, np.kron(ind, Z.T)])  # 6 + 24 features per document


def fit_rank(X_pairs, lam):
    def f(w):
        m = X_pairs @ w
        loss = np.mean(np.logaddexp(0, -m)) + lam * w @ w
        g = -(X_pairs.T @ (1 / (1 + np.exp(m)))) / len(m) + 2 * lam * w
        return loss, g
    return minimize(f, np.zeros(X_pairs.shape[1]), jac=True, method="L-BFGS-B").x


def pair_index(grades):
    return np.argwhere(grades[:, None] > grades[None, :])  # (i, j) with grade_i > grade_j, row-major order


def pairs_for(items, rng):
    X = []
    for feats, cand in items:
        if len(cand) > MAX_PAIRS:
            cand = cand[rng.choice(len(cand), MAX_PAIRS, replace=False)]
        X.append(feats[cand[:, 0]] - feats[cand[:, 1]])
    return np.vstack(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pubmedbert-doc-cache", default=None)
    ap.add_argument("--repro-bm25", type=float, default=0.3103001666796228)
    args = ap.parse_args()
    t0, started = time.time(), datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(args.cache, exist_ok=True)
    R = {"stage": "0 (exploratory pilot)", "fidelity": {}}

    corpus = [json.loads(l) for l in open(os.path.join(args.data, "corpus.jsonl"))]
    qtext = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(os.path.join(args.data, "queries.jsonl"))}
    qrels = {}
    for split in ["train", "dev", "test"]:
        qrels[split] = {}
        with open(os.path.join(args.data, "qrels", f"{split}.tsv")) as f:
            next(f)
            for line in f:
                q, d, g = line.rstrip("\n").split("\t")
                qrels[split].setdefault(q, {})[d] = float(g)
    ids = [r["_id"] for r in corpus]
    texts = [f"{r['title']} {r['text']}" for r in corpus]  # what the repository's BM25 indexes
    splits = {s: sorted(q for q in qrels[s] if q in qtext) for s in qrels}
    log(f"{len(corpus)} docs; queries train/dev/test = {[len(splits[s]) for s in ['train', 'dev', 'test']]}")

    # ---- gate 1: A5 mapping unit tests ----
    tests = mapping_unit_tests()
    R["fidelity"]["a5_mapping_unit_tests"] = {"passed": sum(t["ok"] for t in tests), "of": len(tests), "cases": tests}
    assert all(t["ok"] for t in tests), "A5 mapping unit tests failed"

    # ---- gate 2: A6 sparse BM25 must reproduce the repository's BM25 on the test queries ----
    ref_time = datetime(2020, 1, 1)
    docs_repo = [Document(id=ids[i], title=corpus[i]["title"], abstract=corpus[i]["text"],
                          content_embedding=np.zeros(1), publication_date=ref_time) for i in range(len(corpus))]
    bm_repo = BM25(); bm_repo.index(docs_repo)
    bm = SparseBM25(texts)
    mism, ndcgs = 0, []
    for q in splits["test"]:
        a = [i for i, _ in bm_repo.search(qtext[q], POOL)]
        b, _ = bm.search(qtext[q], POOL)
        mism += a != b
        ndcgs.append(ndcg_at_k([ids[i] for i in b], qrels["test"][q], K))
    gate = {"queries": len(splits["test"]), "top100_mismatches": mism, "ndcg10": float(np.mean(ndcgs)),
            "august": args.repro_bm25, "abs_diff": abs(float(np.mean(ndcgs)) - args.repro_bm25)}
    gate["passed"] = mism == 0 and gate["abs_diff"] < 1e-9
    R["fidelity"]["a6_sparse_bm25_gate"] = gate
    log(f"A6 gate: {gate}")
    assert gate["passed"], "A6 sparse BM25 fidelity gate failed; Stage 0 stops here"

    # ---- embeddings ----
    log("Embedding with MedCPT and (for F) the August encoder")
    art_inputs = [[r["title"], r["text"]] for r in corpus]
    E_art, p_art = embed(art_inputs, "medcpt_article", args.cache, pairs=True)
    q_needed = splits["train"] + splits["dev"]
    E_q, p_q = embed([qtext[q] for q in q_needed], "medcpt_query", args.cache)
    qrow = {q: i for i, q in enumerate(q_needed)}
    pmb_doc_texts = [f"{r['title']} {r['text']}".strip() for r in corpus]
    if args.pubmedbert_doc_cache and sha256_file(args.pubmedbert_doc_cache) == AUDIT_PUBMEDBERT_DOCS_SHA:
        E_pmb, p_pmb = np.load(args.pubmedbert_doc_cache), args.pubmedbert_doc_cache
        pmb_source = "audit cache, sha256 matches the audit manifest"
    else:
        E_pmb, p_pmb = embed(pmb_doc_texts, "pubmedbert", args.cache)
        pmb_source = "computed"
    E_pmb_q, p_pmb_q = embed([qtext[q] for q in splits["dev"]], "pubmedbert", args.cache)
    pmb_qrow = {q: i for i, q in enumerate(splits["dev"])}
    action = np.array([actionability_relevance(d, None) for d in docs_repo])

    def module_scores(q):
        """Pool (BM25 top 100, BM25 order) and the six raw module scores for its documents."""
        pool, s = bm.search(qtext[q], POOL)
        qv = E_q[qrow[q]]
        raw = {
            "bm25": s[pool],
            "topical": np.array([cosine_similarity(E_art[i], qv) for i in pool]),
            "evidence": np.zeros(len(pool)),       # no study-type labels on NFCorpus: pool mean, z = 0
            "temporal": np.zeros(len(pool)),       # no dates on NFCorpus: z = 0
            "applicability": np.zeros(len(pool)),  # off (A12)
            "actionability": action[pool],
        }
        return pool, raw

    def Zmat(raw):
        return np.vstack([z(raw[m]) for m in MODULES])

    def gated(Z, gates):
        return sum((0 if m in GLOBALLY_OFF else gates[m]) * Z[i] for i, m in enumerate(MODULES))

    # ---- L: train on training queries ----
    log("Training L")
    train_items = []
    for q in splits["train"]:
        pool, raw = module_scores(q)
        grades = np.array([qrels["train"][q].get(ids[i], 0.0) for i in pool])
        if grades.max() > grades.min():
            train_items.append((q, l_features(Zmat(raw), question_type(qtext[q])), pair_index(grades), pool))
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(train_items)), 5)
    cv = {}
    for lam in LAMBDAS:
        scores = []
        for k in range(5):
            held = set(folds[k].tolist())
            prng = np.random.default_rng(0)
            Xp = pairs_for([(f, c) for j, (_, f, c, _) in enumerate(train_items) if j not in held], prng)
            w = fit_rank(Xp, lam)
            for j in held:
                q, f, _, pool = train_items[j]
                scores.append(ndcg_at_k(rank_by(f @ w, [ids[i] for i in pool]), qrels["train"][q], K))
        cv[lam] = float(np.mean(scores))
    best_lam = max(LAMBDAS, key=lambda l: (cv[l], -LAMBDAS.index(l)))
    w_L = fit_rank(pairs_for([(f, c) for _, f, c, _ in train_items], np.random.default_rng(0)), best_lam)
    R["L"] = {"training_queries_with_pairs": len(train_items), "cv_mean_ndcg10": {str(k): v for k, v in cv.items()},
              "lambda": best_lam}
    log(f"L: lambda={best_lam}, cv={cv}")

    # ---- dev evaluation ----
    log("Scoring dev queries")
    perm = {s: np.random.default_rng(s).permutation(len(TYPES)) for s in SEEDS}
    R["R_permutations"] = {str(s): [TYPES[i] for i in perm[s]] for s in SEEDS}
    names = ["B0", "F", "M", "U", "V", "L"] + [f"N{s}" for s in SEEDS] + [f"R{s}" for s in SEEDS] + [f"P{s}" for s in SEEDS]
    pq = {n: [] for n in names}
    qtypes = []
    checks = {"m_bm25_only_equals_B0": 0, "V_equals_U": 0, "F_formula_mismatch": 0, "B0_equals_repo_bm25_top100": 0}
    for qi, q in enumerate(splits["dev"]):
        rel = qrels["dev"][q]
        pool, raw = module_scores(q)
        pids = [ids[i] for i in pool]
        t = question_type(qtext[q]); qtypes.append(t)
        Z = Zmat(raw)
        nd = lambda ranked: ndcg_at_k(ranked, rel, K)

        pq["B0"].append(nd(pids))
        checks["B0_equals_repo_bm25_top100"] += [i for i, _ in bm_repo.search(qtext[q], POOL)] == pool
        only_bm25 = {m: (1 if m == "bm25" else 0) for m in MODULES}
        checks["m_bm25_only_equals_B0"] += rank_by(gated(Z, only_bm25), pids) == pids
        sM = gated(Z, GATES[t]); pq["M"].append(nd(rank_by(sM, pids)))
        sU = gated(Z, {m: 1 for m in MODULES}); pq["U"].append(nd(rank_by(sU, pids)))
        sV = gated(Z, {m: (0 if m == "evidence" else 1) for m in MODULES}); pq["V"].append(nd(rank_by(sV, pids)))
        checks["V_equals_U"] += rank_by(sV, pids) == rank_by(sU, pids)

        # F (A7): version 0.1 R(d, q) over the BM25 pool, August encoder, keyword evidence classifier, tau = 1
        qF = Query(text=qtext[q], embedding=E_pmb_q[pmb_qrow[q]])
        dF = [Document(id=ids[i], title=corpus[i]["title"], abstract=corpus[i]["text"],
                       content_embedding=E_pmb[i], publication_date=ref_time) for i in pool]
        sF = [clinical_relevance(d, qF, query_time=ref_time) for d in dF]
        indep = [evidence_weight(d.evidence_level) * (0.5 * cosine_similarity(d.content_embedding, qF.embedding)
                 + 0.3 * 0.5 + 0.2 * actionability_relevance(d, qF)) for d in dF]
        checks["F_formula_mismatch"] += int(max(abs(a - b) for a, b in zip(sF, indep)) > 1e-12)
        pq["F"].append(nd([d.id for d, _ in sorted(zip(dF, sF), key=lambda x: x[1], reverse=True)]))

        pq["L"].append(nd(rank_by(l_features(Z, t) @ w_L, pids)))
        grades = np.array([rel.get(p, 0.0) for p in pids])
        for s in SEEDS:
            nrng = np.random.default_rng([s, qi, 11])
            rawN = dict(raw)
            for m in MODULES:
                if m != "bm25":
                    rawN[m] = nrng.standard_normal(len(pool))
            pq[f"N{s}"].append(nd(rank_by(gated(Zmat(rawN), GATES[t]), pids)))
            tR = TYPES[perm[s][TYPES.index(t)]]
            pq[f"R{s}"].append(nd(rank_by(gated(Z, GATES[tR]), pids)))
            prng = np.random.default_rng([s, qi, 13])
            pq[f"P{s}"].append(nd(rank_by(sM + z(grades + prng.standard_normal(len(pool))), pids)))
        if qi % 50 == 0:
            log(f"  dev query {qi + 1}/{len(splits['dev'])}")
    n_dev = len(splits["dev"])
    checks = {k: (v if k == "F_formula_mismatch" else f"{v}/{n_dev}") for k, v in checks.items()}
    R["fidelity"]["dev_checks"] = checks
    R["question_types_dev"] = dict(Counter(qtypes))

    A = {n: np.asarray(v) for n, v in pq.items()}
    Nbar = np.mean([A[f"N{s}"] for s in SEEDS], 0)
    Rbar = np.mean([A[f"R{s}"] for s in SEEDS], 0)
    Pbar = np.mean([A[f"P{s}"] for s in SEEDS], 0)
    R["means"] = {n: float(v.mean()) for n, v in A.items()}
    R["means"].update({"N_bar": float(Nbar.mean()), "R_bar": float(Rbar.mean()), "P_bar": float(Pbar.mean())})

    primary = paired(A["M"] - A["B0"])
    fam = {"M-F": A["M"] - A["F"], "M-U": A["M"] - A["U"], "M-V": A["M"] - A["V"],
           "M-L": A["M"] - A["L"], "M-Rbar": A["M"] - Rbar}
    sec = {k: paired(v) for k, v in fam.items()}
    for k, p in zip(sec, holm([sec[k]["p"] for k in sec])):
        sec[k]["p_holm"] = p
    controls = {"Nbar-B0": paired(Nbar - A["B0"]), "Pbar-M": paired(Pbar - A["M"])}
    R["primary_M_minus_B0"], R["secondary_holm"], R["controls"] = primary, sec, controls

    sig = lambda k: sec[k]["p_holm"] < 0.05 and sec[k]["mean"] > 0
    if controls["Pbar-M"]["ci95"][0] <= 0:
        verdict = "Test failed"
    elif primary["ci95"][1] < 0.05:
        verdict = "Refuted"
    elif (primary["mean"] >= 0.05 and primary["ci95"][0] > 0 and sig("M-F") and sig("M-U") and sig("M-V")
          and controls["Nbar-B0"]["ci95"][1] < 0.05):
        verdict = "Supported"
    elif primary["mean"] >= 0.05 and primary["ci95"][0] > 0:
        verdict = "Gain without attribution"
    else:
        verdict = "Inconclusive"
    R["decision_rule_applied_exploratory"] = verdict
    R["stage0_predictions"] = {
        "|M-B0| < 0.03": abs(primary["mean"]) < 0.03,
        "|Nbar-B0| < 0.03": abs(controls["Nbar-B0"]["mean"]) < 0.03,
        "Pbar-M > 0 with CI lower > 0": controls["Pbar-M"]["mean"] > 0 and controls["Pbar-M"]["ci95"][0] > 0,
    }
    R["runtime_seconds"] = round(time.time() - t0, 1)

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(R, f, indent=2)
    with open(os.path.join(HERE, "per_query.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["qid", "question_type"] + names)
        for i, q in enumerate(splits["dev"]):
            w.writerow([q, qtypes[i]] + [f"{pq[n][i]:.10f}" for n in names])
    import torch, transformers, scipy
    manifest = {
        "stage": "0", "preregistration": "../PREREGISTRATION.md", "amendment_commit": "4dbc7ea",
        "started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "data": {"source": "BeIR/nfcorpus rev b5026a0e96e8a7ac4f95f482a596389289d46269; "
                           "BeIR/nfcorpus-qrels rev a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
                 "sha256": {p: sha256_file(os.path.join(args.data, p)) for p in
                            ["corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/dev.tsv", "qrels/test.tsv"]}},
        "encoders": {k: {"name": v[0], "revision": v[1], "pooling": v[2], "max_length": v[3]} for k, v in ENCODERS.items()},
        "pubmedbert_doc_embeddings": pmb_source,
        "embeddings_sha256": {os.path.basename(p): sha256_file(p) for p in [p_art, p_q, p_pmb, p_pmb_q]},
        "seeds": {"N/R/P": SEEDS, "L folds and pairs": 0, "bootstrap and sign-flip": 0},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(),
                        "packages": {"numpy": np.__version__, "scipy": scipy.__version__,
                                     "torch": torch.__version__, "transformers": transformers.__version__}},
    }
    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log(json.dumps({k: R[k] for k in ["means", "primary_M_minus_B0", "controls", "decision_rule_applied_exploratory",
                                       "stage0_predictions"]}, indent=1))
    log("done")


if __name__ == "__main__":
    main()

"""
The topical component alone, scored with a similarity-trained encoder (MedCPT), against
BM25 on BEIR/NFCorpus. See PREREGISTRATION.md in this directory, committed before this ran.

The framework is imported unchanged: topical_relevance (cosine), BM25, clinical_relevance
and ndcg_at_k all come from clinical_ir. This script supplies only the encoders, which no
code in the package loads.

    pip install -e . torch transformers
    HF_HUB_DISABLE_XET=1 python evals/topical-2026-10/run_topical.py \
        --data datasets/nfcorpus --cache .cache
"""
import argparse, csv, hashlib, json, os, platform, sys, time
from datetime import datetime, timezone
import numpy as np

from clinical_ir.relevance import Document, Query, topical_relevance, clinical_relevance, \
    applicability_relevance, actionability_relevance, COMPONENT_WEIGHTS
from clinical_ir.evidence import evidence_weight
from clinical_ir.retrieval import BM25
from clinical_ir.evaluate import ndcg_at_k

HERE = os.path.dirname(os.path.abspath(__file__))

MEDCPT_Q = ("ncbi/MedCPT-Query-Encoder", "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc")
MEDCPT_A = ("ncbi/MedCPT-Article-Encoder", "d05a736da4bb84ee4057b7f7999485be6ed85465")
PUBMEDBERT = ("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
              "e1354b7a3a09615f6aba48dfad4b7a613eef7062")

K, CAND_K = 10, 100
N_RAND = 200          # random reorders / tie-breaks per query
N_EMB_SEEDS = 5       # random-vector encoder controls
N_BOOT = N_PERM = 10_000
MARGIN = 0.03
REF_TIME = datetime(2020, 1, 1)   # publication_date == query_time, so tau == 1 (as August)


# ---------- data ----------
def load(data):
    corpus = [json.loads(l) for l in open(os.path.join(data, "corpus.jsonl"))]
    qrels = {}
    with open(os.path.join(data, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            qid, did, s = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = float(s)
    qtext = {r["_id"]: r["text"] for r in map(json.loads, open(os.path.join(data, "queries.jsonl")))}
    queries = sorted((qid, qtext[qid]) for qid in qrels if qid in qtext)
    return corpus, queries, qrels


# ---------- encoders ----------
def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def encode(model, inputs, max_length, pooling, cache, tag):
    """inputs: list of str, or list of [str, str] sentence pairs. Cached on model, revision
    and a hash of the exact inputs."""
    name, rev = model
    key = sha(json.dumps([name, rev, max_length, pooling, inputs]))[:16]
    path = os.path.join(cache, f"emb_{tag}_{key}.npy")
    if os.path.exists(path):
        return np.load(path), path
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    mdl = AutoModel.from_pretrained(name, revision=rev).eval()
    # Encode in length order so padding stays small; restore the original order after.
    lens = [len(x if isinstance(x, str) else x[0] + " " + x[1]) for x in inputs]
    order = np.argsort(lens, kind="stable")
    out = np.zeros((len(inputs), mdl.config.hidden_size), dtype=np.float32)
    bs, t0 = 16, time.time()
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        batch = [inputs[i] for i in idx]
        if isinstance(batch[0], str):
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        else:
            enc = tok([p[0] for p in batch], [p[1] for p in batch], padding=True,
                      truncation="longest_first", max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            hs = mdl(**enc).last_hidden_state
        if pooling == "cls":
            v = hs[:, 0, :]
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out[idx] = v.numpy().astype(np.float32)
        if (b // bs) % 25 == 0:
            print(f"  [{tag}] {b + len(idx)}/{len(inputs)}  {time.time() - t0:.0f}s", flush=True)
    np.save(path, out)
    return out, path


# ---------- statistics (same procedure as audit-2026-10-03) ----------
def paired(diffs, seed=0):
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = np.abs((signs * d).mean(1))
    p = (1 + np.sum(null >= abs(d.mean()) - 1e-15)) / (N_PERM + 1)
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "ci95": [float(lo), float(hi)],
            "p_signflip": float(p), "n": int(len(d)),
            "mde80": float(2.8 * d.std(ddof=1) / np.sqrt(len(d))),
            "favour_first": int((d > 0).sum()), "favour_second": int((d < 0).sum()),
            "ties": int((d == 0).sum())}


def primary_verdict(ci):
    lo, hi = ci
    if -MARGIN <= lo and hi <= MARGIN:
        return "equivalent"
    if lo > 0:
        return "topical beats BM25"
    if hi < 0:
        return "topical loses to BM25"
    return "inconclusive"


# ---------- ranking helpers ----------
def rank_desc(idx_scores):
    """Stable descending sort: ties keep their input order, as the framework's sorts do."""
    return sorted(idx_scores, key=lambda x: x[1], reverse=True)


def ndcg_of(ranked, docs, rel):
    return ndcg_at_k([docs[i].id for i, _ in ranked[:K]], rel, K)


def random_ties(pool_scores, docs, rel, rng):
    """Shuffle, then the same stable sort: expected NDCG@10 under random tie-breaking."""
    vals = []
    for _ in range(N_RAND):
        perm = rng.permutation(len(pool_scores))
        vals.append(ndcg_of(rank_desc([pool_scores[j] for j in perm]), docs, rel))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    corpus, queries, qrels = load(args.data)
    n_judg = sum(len(v) for v in qrels.values())
    print(f"{len(corpus)} docs, {len(queries)} test queries, {n_judg} judgments", flush=True)
    assert (len(corpus), len(queries), n_judg) == (3633, 323, 12334), "data does not match preregistration"

    # ---- encoders ----
    q_texts = [t for _, t in queries]
    print("MedCPT query encoder...", flush=True)
    q_med, q_med_path = encode(MEDCPT_Q, q_texts, 64, "cls", args.cache, "medcpt_q")
    print("MedCPT article encoder...", flush=True)
    d_med, d_med_path = encode(MEDCPT_A, [[r["title"], r["text"]] for r in corpus], 512, "cls",
                               args.cache, "medcpt_a")
    print("PubMedBERT mean-pooled (S2, the August encoder)...", flush=True)
    d_pmb, d_pmb_path = encode(PUBMEDBERT, [f"{r['title']} {r['text']}".strip() for r in corpus], 512,
                               "mean", args.cache, "pubmedbert_docs")
    q_pmb, q_pmb_path = encode(PUBMEDBERT, q_texts, 512, "mean", args.cache, "pubmedbert_queries")

    def make_docs(emb):
        return [Document(id=r["_id"], title=r["title"], abstract=r["text"], content_embedding=emb[i],
                         publication_date=REF_TIME, population=None) for i, r in enumerate(corpus)]

    docs_med = make_docs(d_med)
    docs_pmb = make_docs(d_pmb)
    bm25 = BM25()
    bm25.index(docs_med)

    rand_docs, rand_q = [], []
    for s in range(N_EMB_SEEDS):
        rng = np.random.default_rng(s)
        rand_docs.append(make_docs(rng.standard_normal(d_med.shape).astype(np.float32)))
        rand_q.append(rng.standard_normal(q_med.shape).astype(np.float32))

    # Fidelity: the vectorised cosine used for the random controls must equal topical_relevance.
    def cos_all(D, qv):
        return (D @ qv) / (np.linalg.norm(D, axis=1) * np.linalg.norm(qv))
    q0 = Query(text=queries[0][1], embedding=q_med[0])
    ref = np.array([topical_relevance(d, q0) for d in docs_med])
    assert np.allclose(cos_all(d_med, q_med[0]), ref, atol=1e-6), "vectorised cosine mismatch"
    D_rand = [np.stack([d.content_embedding for d in rd]) for rd in rand_docs]

    cols = ["bm25", "medcpt_cos", "pubmedbert_cos"] + [f"random_vec_seed{s}" for s in range(N_EMB_SEEDS)] + \
           ["medcpt_dot", "bm25_pool_by_bm25", "pool_medcpt_cos", "pool_random", "composite_medcpt",
            "composite_oracle", "composite_oracle_random_ties", "oracle_ceiling"]
    PQ = {c: [] for c in cols}
    meta = {"n_bm25_nonzero": [], "distinct_in_pool": {c: [] for c in
            ["bm25_pool_by_bm25", "pool_medcpt_cos", "composite_medcpt", "composite_oracle"]},
            "tie_at_10": {c: 0 for c in ["bm25", "medcpt_cos", "pubmedbert_cos", "medcpt_dot"]}}
    t0 = time.time()
    for qi, (qid, qtext) in enumerate(queries):
        rel = qrels[qid]
        rng = np.random.default_rng([qi, 7])

        bm_scores = [bm25.score(qtext, i) for i in range(len(corpus))]
        meta["n_bm25_nonzero"].append(int(sum(s > 0 for s in bm_scores)))
        bm_rank = rank_desc(list(enumerate(bm_scores)))
        if qi == 0:
            assert [i for i, _ in bm_rank[:CAND_K]] == [i for i, _ in bm25.search(qtext, CAND_K)]
        PQ["bm25"].append(ndcg_of(bm_rank, docs_med, rel))
        pool = bm_rank[:CAND_K]

        # Primary: full-corpus cosine with MedCPT, through the framework's topical_relevance.
        qm = Query(text=qtext, embedding=q_med[qi])
        med = rank_desc([(i, topical_relevance(d, qm)) for i, d in enumerate(docs_med)])
        PQ["medcpt_cos"].append(ndcg_of(med, docs_med, rel))

        qp = Query(text=qtext, embedding=q_pmb[qi])
        pmb = rank_desc([(i, topical_relevance(d, qp)) for i, d in enumerate(docs_pmb)])
        PQ["pubmedbert_cos"].append(ndcg_of(pmb, docs_pmb, rel))

        for s in range(N_EMB_SEEDS):
            r = rank_desc(list(enumerate(cos_all(D_rand[s], rand_q[s][qi]).tolist())))
            PQ[f"random_vec_seed{s}"].append(ndcg_of(r, docs_med, rel))

        dot = rank_desc(list(enumerate((d_med @ q_med[qi]).tolist())))
        PQ["medcpt_dot"].append(ndcg_of(dot, docs_med, rel))

        for name, ranked in [("bm25", bm_rank), ("medcpt_cos", med), ("pubmedbert_cos", pmb), ("medcpt_dot", dot)]:
            meta["tie_at_10"][name] += int(ranked[K - 1][1] == ranked[K][1])

        # Pool-based checks on BM25's top 100.
        by_bm25 = rank_desc(pool)
        PQ["bm25_pool_by_bm25"].append(ndcg_of(by_bm25, docs_med, rel))
        pool_cos = [(i, topical_relevance(docs_med[i], qm)) for i, _ in pool]
        PQ["pool_medcpt_cos"].append(ndcg_of(rank_desc(pool_cos), docs_med, rel))
        PQ["pool_random"].append(float(np.mean(
            [ndcg_of([(pool[j][0], 0) for j in rng.permutation(CAND_K)], docs_med, rel) for _ in range(N_RAND)])))
        comp = [(i, clinical_relevance(docs_med[i], qm, query_time=REF_TIME)) for i, _ in pool]
        PQ["composite_medcpt"].append(ndcg_of(rank_desc(comp), docs_med, rel))
        W = COMPONENT_WEIGHTS
        orc = [(i, evidence_weight(docs_med[i].evidence_level) *
                (W["topic"] * rel.get(docs_med[i].id, 0) / 2.0 + W["apply"] * applicability_relevance(docs_med[i], qm)
                 + W["action"] * actionability_relevance(docs_med[i], qm))) for i, _ in pool]
        PQ["composite_oracle"].append(ndcg_of(rank_desc(orc), docs_med, rel))
        PQ["composite_oracle_random_ties"].append(random_ties(orc, docs_med, rel, rng))
        PQ["oracle_ceiling"].append(ndcg_of(rank_desc([(i, rel.get(docs_med[i].id, 0)) for i, _ in pool]), docs_med, rel))
        for name, sc in [("bm25_pool_by_bm25", pool), ("pool_medcpt_cos", pool_cos),
                         ("composite_medcpt", comp), ("composite_oracle", orc)]:
            meta["distinct_in_pool"][name].append(len({round(s, 12) for _, s in sc}))

        if qi % 40 == 0:
            print(f"  query {qi + 1}/{len(queries)}  {time.time() - t0:.0f}s", flush=True)

    A = {c: np.array(v) for c, v in PQ.items()}
    means = {c: float(v.mean()) for c, v in A.items()}
    nz = np.array(meta["n_bm25_nonzero"])
    full_pool = nz >= CAND_K

    # ---- primary ----
    prim = paired(A["medcpt_cos"] - A["bm25"])
    prim["verdict"] = primary_verdict(prim["ci95"])
    bm = means["bm25"]
    five = {"absolute_bar": bm + 0.05, "relative_bar": bm * 1.05,
            "meets_absolute": means["medcpt_cos"] >= bm + 0.05,
            "meets_relative": means["medcpt_cos"] >= bm * 1.05,
            "relative_change_pct": 100 * (means["medcpt_cos"] - bm) / bm}

    # ---- controls ----
    rand_means = [means[f"random_vec_seed{s}"] for s in range(N_EMB_SEEDS)]
    controls = {
        "bm25_reproduces_august": {"value": bm, "august": 0.3103001667, "pass": abs(bm - 0.3103001667) <= 0.001},
        "positive_bm25_pool_reorder": {"value": means["bm25_pool_by_bm25"],
                                       "pass": abs(means["bm25_pool_by_bm25"] - bm) <= 0.001},
        "negative_random_vectors": {"seed_means": rand_means,
                                    "any_within_0.05_of_medcpt": any(abs(m - means["medcpt_cos"]) <= 0.05 for m in rand_means)},
        "median_distinct_scores_per_pool": {c: float(np.median(v)) for c, v in meta["distinct_in_pool"].items()},
        "queries_with_tie_at_rank_10": meta["tie_at_10"],
        "oracle_composite": {"stable": means["composite_oracle"], "random_ties": means["composite_oracle_random_ties"],
                             "ceiling": means["oracle_ceiling"]},
    }

    # ---- secondary and exploratory ----
    secondary = {
        "S1_pool_cos_vs_random_queries_with_100_matches": paired(A["pool_medcpt_cos"][full_pool] - A["pool_random"][full_pool]),
        "S1_n_queries": int(full_pool.sum()),
        "S2_medcpt_minus_pubmedbert": paired(A["medcpt_cos"] - A["pubmedbert_cos"]),
        "S2_pubmedbert_minus_bm25": paired(A["pubmedbert_cos"] - A["bm25"]),
        "S3_composite_medcpt_minus_bm25": paired(A["composite_medcpt"] - A["bm25"]),
        "S3_composite_medcpt_minus_medcpt_cos": paired(A["composite_medcpt"] - A["medcpt_cos"]),
    }
    strata = {"lt10": nz < 10, "10to99": (nz >= 10) & (nz < CAND_K), "ge100": full_pool}
    exploratory = {
        "dot_minus_cos": paired(A["medcpt_dot"] - A["medcpt_cos"]),
        "dot_minus_bm25": paired(A["medcpt_dot"] - A["bm25"]),
        "by_bm25_matches": {k: {"n": int(m.sum()), "bm25": float(A["bm25"][m].mean()) if m.any() else None,
                                "medcpt_cos": float(A["medcpt_cos"][m].mean()) if m.any() else None,
                                "diff": paired(A["medcpt_cos"][m] - A["bm25"][m]) if m.sum() > 1 else None}
                            for k, m in strata.items()},
    }

    results = {"means": means, "primary_medcpt_cos_minus_bm25": prim, "five_percent_bar": five,
               "controls": controls, "secondary": secondary, "exploratory": exploratory}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(HERE, "per_query.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "n_bm25_nonzero"] + cols)
        for qi, (qid, _) in enumerate(queries):
            w.writerow([qid, nz[qi]] + [f"{A[c][qi]:.10f}" for c in cols])

    import torch, transformers, tokenizers, scipy
    def fsha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    manifest = {
        "test": "topical-2026-10", "preregistration_commit": "a20b856", "started": started,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": fsha(os.path.abspath(__file__)),
        "data": {"BeIR/nfcorpus revision": "b5026a0e96e8a7ac4f95f482a596389289d46269",
                 "BeIR/nfcorpus-qrels revision": "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
                 "sha256": {n: fsha(os.path.join(args.data, n)) for n in ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]}},
        "encoders": {"medcpt_query": {"name": MEDCPT_Q[0], "revision": MEDCPT_Q[1], "pooling": "cls", "max_length": 64},
                     "medcpt_article": {"name": MEDCPT_A[0], "revision": MEDCPT_A[1], "pooling": "cls",
                                        "max_length": 512, "input": "[title, text] sentence pair"},
                     "pubmedbert_S2": {"name": PUBMEDBERT[0], "revision": PUBMEDBERT[1], "pooling": "mean",
                                       "max_length": 512, "input": "title + ' ' + text"},
                     "device": "cpu", "dtype": "float32"},
        "embeddings_sha256": {os.path.basename(p): fsha(p) for p in [q_med_path, d_med_path, d_pmb_path, q_pmb_path]},
        "seeds": {"bootstrap_and_signflip": 0, "random_reorder_and_ties": "numpy default_rng([query_index, 7]), 200 draws",
                  "random_vectors": list(range(N_EMB_SEEDS))},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(),
                        "packages": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
                                     "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}},
        "command": "HF_HUB_DISABLE_XET=1 python evals/topical-2026-10/run_topical.py --data datasets/nfcorpus --cache .cache",
    }
    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({"means": means, "primary": prim, "five_percent_bar": five,
                      "controls": controls}, indent=2), flush=True)


if __name__ == "__main__":
    main()

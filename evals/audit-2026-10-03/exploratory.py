"""
EXPLORATORY follow-ups, written after the preregistered diagnostics (audit.py) had been
seen. Logged as deviations in PREREGISTRATION.md. Nothing here was predicted in advance.

E1. Why is the hybrid pool's overlap with BM25's top 100 only ~0.67? Count, per query,
    the documents with a non-zero BM25 score.
E2. Can the composite R(d,q) carry a perfect topical signal? Replace r_topic with the gold
    grade (grade / 2, so it lies in [0, 1]) for every pool document, keep every other factor
    as specified, and rerank the same pool. If R with an oracle topical score still falls
    far below the oracle ceiling, a better encoder cannot rescue the composite.

    PYTHONPATH=<dir with clinical_ir symlink> python evals/audit-2026-10-03/exploratory.py \
        --data <nfcorpus dir> --cache <embedding cache>
"""
import argparse, importlib.util, json, os
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

from clinical_ir.relevance import Document, Query, COMPONENT_WEIGHTS, applicability_relevance, actionability_relevance
from clinical_ir.evidence import evidence_weight
from clinical_ir.retrieval import BM25, hybrid_search
from clinical_ir.evaluate import ndcg_at_k

K, CAND_K = 10, 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("run_eval", os.path.join(REPO, "evals", "run_eval.py"))
    rv = importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)
    rv.DATA, rv.CACHE = args.data, args.cache
    corpus = rv.load_corpus()
    queries, qrels = rv.load_queries_and_qrels()
    doc_emb = rv.embed_texts([f"{t} {x}".strip() for _, t, x in corpus], "docs")
    q_emb = rv.embed_texts([t for _, t in queries], "queries")
    ref = datetime(2020, 1, 1)
    docs = [Document(id=corpus[i][0], title=corpus[i][1], abstract=corpus[i][2],
                     content_embedding=doc_emb[i], publication_date=ref, population=None)
            for i in range(len(corpus))]
    bm25 = BM25(); bm25.index(docs)
    W = COMPONENT_WEIGHTS

    nonzero, overlap = [], []
    ndcg = {"oracle_topic_in_R": [], "oracle_topic_evidence_off": [], "oracle_topic_evidence_action_off": [],
            "oracle_ceiling": []}
    for qi, (qid, qtext) in enumerate(queries):
        rel = qrels[qid]
        q = Query(text=qtext, embedding=q_emb[qi], population=None)
        scores = [bm25.score(qtext, i) for i in range(len(docs))]
        nonzero.append(int(sum(s > 0 for s in scores)))
        bm_ids = {docs[i].id for i, _ in bm25.search(qtext, CAND_K)}
        pool = [i for i, _ in hybrid_search(q, docs, bm25, k=CAND_K)]
        overlap.append(len(bm_ids & {docs[i].id for i in pool}) / CAND_K)

        def rank(use_we, use_action):
            def R(i):
                d = docs[i]
                topic = rel.get(d.id, 0) / 2.0  # oracle topical score in [0, 1]
                comp = (W["topic"] * topic + W["apply"] * applicability_relevance(d, q)
                        + (W["action"] * actionability_relevance(d, q) if use_action else 0.0))
                return (evidence_weight(d.evidence_level) if use_we else 1.0) * comp  # tau == 1 here
            return [docs[i].id for i in sorted(pool, key=R, reverse=True)]
        ndcg["oracle_topic_in_R"].append(ndcg_at_k(rank(True, True), rel, K))
        ndcg["oracle_topic_evidence_off"].append(ndcg_at_k(rank(False, True), rel, K))
        ndcg["oracle_topic_evidence_action_off"].append(ndcg_at_k(rank(False, False), rel, K))
        ndcg["oracle_ceiling"].append(ndcg_at_k(sorted([docs[i].id for i in pool], key=lambda d: rel.get(d, 0), reverse=True), rel, K))

    nonzero, overlap = np.array(nonzero), np.array(overlap)
    out = {
        "E1": {
            "median_docs_with_nonzero_bm25": float(np.median(nonzero)),
            "queries_with_fewer_than_100_nonzero": int((nonzero < CAND_K).sum()),
            "queries_with_zero_nonzero": int((nonzero == 0).sum()),
            "mean_overlap_when_>=100_nonzero": float(overlap[nonzero >= CAND_K].mean()),
            "mean_overlap_when_<100_nonzero": float(overlap[nonzero < CAND_K].mean()),
            "n_queries": int(len(nonzero)),
        },
        "E2": {k: float(np.mean(v)) for k, v in ndcg.items()},
    }
    with open(os.path.join(HERE, "exploratory_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

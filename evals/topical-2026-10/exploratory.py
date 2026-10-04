"""
Exploratory, written after seeing results.json (see the deviation log in PREREGISTRATION.md).

Why does the composite R(d, q) collapse with MedCPT (S3: 0.088) when MedCPT cosine alone ranks
the same pool at 0.324? Per query, over BM25's top 100: the spread of each additive term and of
the evidence weight, and the Spearman correlation of the final score with each factor.

    python evals/topical-2026-10/exploratory.py --data datasets/nfcorpus --cache .cache
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import spearmanr

from clinical_ir.relevance import Document, Query, topical_relevance, clinical_relevance, \
    actionability_relevance, COMPONENT_WEIGHTS
from clinical_ir.evidence import evidence_weight
from clinical_ir.retrieval import BM25

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("rt", os.path.join(HERE, "run_topical.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    corpus, queries, qrels = rt.load(a.data)
    q_med, _ = rt.encode(rt.MEDCPT_Q, [t for _, t in queries], 64, "cls", a.cache, "medcpt_q")
    d_med, _ = rt.encode(rt.MEDCPT_A, [[r["title"], r["text"]] for r in corpus], 512, "cls", a.cache, "medcpt_a")
    docs = [Document(id=r["_id"], title=r["title"], abstract=r["text"], content_embedding=d_med[i],
                     publication_date=rt.REF_TIME) for i, r in enumerate(corpus)]
    bm25 = BM25(); bm25.index(docs)
    W = COMPONENT_WEIGHTS
    sd_topic, sd_action, sd_we, rho = [], [], [], {"w_e": [], "cos": [], "action": []}
    cos_range = []
    for qi, (qid, qtext) in enumerate(queries):
        q = Query(text=qtext, embedding=q_med[qi])
        pool = [i for i, _ in rt.rank_desc([(i, bm25.score(qtext, i)) for i in range(len(docs))])[:rt.CAND_K]]
        cos = np.array([topical_relevance(docs[i], q) for i in pool])
        act = np.array([actionability_relevance(docs[i], q) for i in pool])
        we = np.array([evidence_weight(docs[i].evidence_level) for i in pool])
        R = np.array([clinical_relevance(docs[i], q, query_time=rt.REF_TIME) for i in pool])
        sd_topic.append((W["topic"] * cos).std()); sd_action.append((W["action"] * act).std()); sd_we.append(we.std())
        cos_range.append((cos.min(), cos.max()))
        for k, v in [("w_e", we), ("cos", cos), ("action", act)]:
            r = spearmanr(R, v).correlation
            if np.isfinite(r):
                rho[k].append(r)
    cr = np.array(cos_range)
    out = {
        "median_sd_within_pool": {"0.5*cos": float(np.median(sd_topic)), "0.2*actionability": float(np.median(sd_action)),
                                  "w_e": float(np.median(sd_we))},
        "median_cos_min_max_in_pool": [float(np.median(cr[:, 0])), float(np.median(cr[:, 1]))],
        "median_spearman_final_score_with": {k: float(np.median(v)) for k, v in rho.items()},
    }
    with open(os.path.join(HERE, "exploratory_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

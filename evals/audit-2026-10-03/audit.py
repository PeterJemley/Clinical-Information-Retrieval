"""
Diagnostics for the 23 August NFCorpus evaluation. See PREREGISTRATION.md in this
directory for the claims, predictions and decision rules, committed before this ran.

The August harness (evals/run_eval.py) is imported, not copied, so its loaders, encoder
and scorer are the ones diagnosed. Before any diagnostic, every August configuration is
re-derived here and must match the reproduction run's results.json to 1e-9.

    PYTHONPATH=<dir containing a clinical_ir symlink to the repo> \
    python evals/audit-2026-10-03/audit.py --data <nfcorpus dir> --cache <embedding cache> \
        --repro-results <results.json from the reproduction run>
"""
import argparse, csv, importlib.util, json, os, sys, time
from datetime import datetime
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

from clinical_ir.relevance import Document, Query, COMPONENT_WEIGHTS
from clinical_ir.evidence import evidence_weight
from clinical_ir.relevance import topical_relevance, actionability_relevance
from clinical_ir.retrieval import BM25, hybrid_search, z_normalize, HYBRID_BETA
from clinical_ir.evaluate import ndcg_at_k, precision_at_k

N_RAND = 200          # random reranks / tie-breaks per query
N_EMB_SEEDS = 5       # random-embedding controls
N_BOOT = 10_000
N_PERM = 10_000
MARGIN = 0.03         # equivalence margin for D1 and D4b
K, CAND_K = 10, 100


def load_august_harness():
    spec = importlib.util.spec_from_file_location("run_eval", os.path.join(REPO, "evals", "run_eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- paired statistics ----------
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
            "favour_first": int((d > 0).sum()), "favour_second": int((d < 0).sum()), "ties": int((d == 0).sum())}


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def equivalence_verdict(ci, margin=MARGIN):
    lo, hi = ci
    if -margin <= lo and hi <= margin:
        return "at chance (within margin)"
    if lo > 0:
        return "better than chance"
    if hi < 0:
        return "worse than chance"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repro-results", required=True)
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    t0 = time.time()

    rv = load_august_harness()
    rv.DATA, rv.CACHE = args.data, args.cache
    corpus = rv.load_corpus()
    queries, qrels = rv.load_queries_and_qrels()
    doc_emb = rv.embed_texts([f"{t} {x}".strip() for _, t, x in corpus], "docs")
    q_emb = rv.embed_texts([t for _, t in queries], "queries")
    print(f"{len(corpus)} docs, {len(queries)} queries, emb {doc_emb.shape} {q_emb.shape}", flush=True)

    ref_time = datetime(2020, 1, 1)
    documents = [Document(id=corpus[i][0], title=corpus[i][1], abstract=corpus[i][2],
                          content_embedding=doc_emb[i], publication_date=ref_time, population=None)
                 for i in range(len(corpus))]
    bm25 = BM25(); bm25.index(documents)
    doc_ids = [d.id for d in documents]
    w_e_all = np.array([evidence_weight(d.evidence_level) for d in documents])
    action_all = np.array([actionability_relevance(d, None) for d in documents])
    dn = doc_emb / np.linalg.norm(doc_emb, axis=1, keepdims=True)

    configs = ["bm25", "full_clinical", "abl_temporal_off", "abl_evidence_off",
               "abl_topic_off", "abl_apply_off", "abl_action_off"]
    pq = {c: [] for c in configs}
    extra = {k: [] for k in ["random_rerank", "topic_off_random_ties", "cos_only_rerank",
                             "bm25_rerank_of_pool", "oracle_rerank_of_pool", "hybrid_unreranked",
                             "p10_bm25", "p10_full"]}
    diag = {k: [] for k in ["sd_ratio", "overlap_bm25_top100", "pool_recall_judged",
                            "bm25_top10_in_pool", "distinct_topic_off_scores",
                            "rho_we", "rho_cos", "rho_action"]}
    rho_nan = {"rho_we": 0, "rho_cos": 0, "rho_action": 0}

    def ids_of(ranked):
        return [doc_ids[i] for i, _ in ranked]

    for qi, (qid, qtext) in enumerate(queries):
        query = Query(text=qtext, embedding=q_emb[qi], population=None)
        rel = qrels[qid]
        relevant = {d for d, g in rel.items() if g > 0}

        # --- the seven August configurations, through the August code path ---
        bm25_top = bm25.search(qtext, CAND_K)
        bm25_ids = [doc_ids[i] for i, _ in bm25_top]
        pq["bm25"].append(ndcg_at_k(bm25_ids, rel, K))
        cands = hybrid_search(query, documents, bm25, k=CAND_K)
        full_ranked = rv.rerank(cands, documents, query, COMPONENT_WEIGHTS, True, True, ref_time)
        pq["full_clinical"].append(ndcg_at_k(ids_of(full_ranked), rel, K))
        pq["abl_temporal_off"].append(ndcg_at_k(ids_of(rv.rerank(cands, documents, query, COMPONENT_WEIGHTS, False, True, ref_time)), rel, K))
        pq["abl_evidence_off"].append(ndcg_at_k(ids_of(rv.rerank(cands, documents, query, COMPONENT_WEIGHTS, True, False, ref_time)), rel, K))
        topic_off = rv.rerank(cands, documents, query, rv.w_without("topic"), True, True, ref_time)
        pq["abl_topic_off"].append(ndcg_at_k(ids_of(topic_off), rel, K))
        pq["abl_apply_off"].append(ndcg_at_k(ids_of(rv.rerank(cands, documents, query, rv.w_without("apply"), True, True, ref_time)), rel, K))
        pq["abl_action_off"].append(ndcg_at_k(ids_of(rv.rerank(cands, documents, query, rv.w_without("action"), True, True, ref_time)), rel, K))

        pool = [i for i, _ in cands]
        pool_ids = [doc_ids[i] for i in pool]
        rng = np.random.default_rng([qi, 7])

        # D1: random reordering of the same pool
        extra["random_rerank"].append(float(np.mean(
            [ndcg_at_k([pool_ids[j] for j in rng.permutation(CAND_K)], rel, K) for _ in range(N_RAND)])))

        # D3: topic-off with random tie-breaking (shuffle, then the same stable sort)
        to_score = {i: s for i, s in topic_off}
        vals = []
        for _ in range(N_RAND):
            shuffled = [pool[j] for j in rng.permutation(CAND_K)]
            ranked = sorted(shuffled, key=lambda i: to_score[i], reverse=True)
            vals.append(ndcg_at_k([doc_ids[i] for i in ranked], rel, K))
        extra["topic_off_random_ties"].append(float(np.mean(vals)))
        diag["distinct_topic_off_scores"].append(len({round(s, 12) for s in to_score.values()}))

        # D4a: cosine-only rerank of the pool
        cos_pool = {i: topical_relevance(documents[i], query) for i in pool}
        extra["cos_only_rerank"].append(ndcg_at_k([doc_ids[i] for i in sorted(pool, key=lambda i: cos_pool[i], reverse=True)], rel, K))

        # D6: positive control and ceiling
        b_pool = {i: bm25.score(qtext, i) for i in pool}
        extra["bm25_rerank_of_pool"].append(ndcg_at_k([doc_ids[i] for i in sorted(pool, key=lambda i: b_pool[i], reverse=True)], rel, K))
        extra["oracle_rerank_of_pool"].append(ndcg_at_k(sorted(pool_ids, key=lambda d: rel.get(d, 0), reverse=True), rel, K))
        extra["hybrid_unreranked"].append(ndcg_at_k(pool_ids, rel, K))
        diag["pool_recall_judged"].append(len(set(pool_ids) & relevant) / len(relevant))
        diag["bm25_top10_in_pool"].append(len(set(bm25_ids[:K]) & set(pool_ids)) / K)
        diag["overlap_bm25_top100"].append(len(set(bm25_ids) & set(pool_ids)) / CAND_K)

        # D2: spread of each term of the hybrid score over the whole corpus
        z = np.array(z_normalize([bm25.score(qtext, i) for i in range(len(documents))]))
        cos_all = dn @ (q_emb[qi] / np.linalg.norm(q_emb[qi]))
        diag["sd_ratio"].append(float(np.std(HYBRID_BETA * z) / np.std((1 - HYBRID_BETA) * cos_all)))

        # D5: which factor orders the full clinical score inside the pool
        full_s = np.array([s for _, s in sorted(full_ranked, key=lambda x: pool.index(x[0]))])
        for key, x in [("rho_we", w_e_all[pool]), ("rho_cos", np.array([cos_pool[i] for i in pool])),
                       ("rho_action", action_all[pool])]:
            r = spearmanr(full_s, x).statistic if np.ptp(x) > 0 else np.nan
            if np.isnan(r):
                rho_nan[key] += 1
            else:
                diag[key].append(float(r))

        # D8: precision@10 (relevant = any positive grade)
        extra["p10_bm25"].append(precision_at_k(bm25_ids, relevant, K))
        extra["p10_full"].append(precision_at_k(ids_of(full_ranked), relevant, K))

        if qi % 50 == 0:
            print(f"  query {qi+1}/{len(queries)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------- fidelity: re-derived August configs must equal the reproduction run ----------
    repro = json.load(open(args.repro_results))
    fidelity = {}
    for c in configs:
        mine, theirs = float(np.mean(pq[c])), repro["configs"][c]["ndcg@10"]
        fidelity[c] = {"audit": mine, "repro": theirs, "abs_diff": abs(mine - theirs)}
        assert abs(mine - theirs) < 1e-9, f"fidelity failure on {c}: {mine} vs {theirs}"
    print("Fidelity check OK: all seven August configurations re-derived exactly", flush=True)

    # ---------- D4b: random-embedding controls, full pipeline ----------
    rand_emb = []
    for seed in range(N_EMB_SEEDS):
        rng = np.random.default_rng(seed)
        de = rng.standard_normal(doc_emb.shape).astype(np.float32)
        qe = rng.standard_normal(q_emb.shape).astype(np.float32)
        docs_r = [Document(id=corpus[i][0], title=corpus[i][1], abstract=corpus[i][2],
                           content_embedding=de[i], publication_date=ref_time, population=None)
                  for i in range(len(corpus))]
        vals = []
        for qi, (qid, qtext) in enumerate(queries):
            qr = Query(text=qtext, embedding=qe[qi], population=None)
            cands = hybrid_search(qr, docs_r, bm25, k=CAND_K)
            vals.append(ndcg_at_k(ids_of(rv.rerank(cands, docs_r, qr, COMPONENT_WEIGHTS, True, True, ref_time)), qrels[qid], K))
        rand_emb.append(vals)
        print(f"  random-embedding seed {seed}: NDCG@10 = {np.mean(vals):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    rand_emb = np.array(rand_emb)

    # ---------- analysis ----------
    full = np.array(pq["full_clinical"])
    R = {"fidelity": fidelity, "n_queries": len(queries)}
    R["means"] = {k: float(np.mean(v)) for k, v in {**pq, **extra}.items()}
    R["means"].update({f"random_embedding_seed{s}": float(rand_emb[s].mean()) for s in range(N_EMB_SEEDS)})

    d1 = paired(full - np.array(extra["random_rerank"]))
    d1["verdict"] = equivalence_verdict(d1["ci95"])
    R["D1_full_vs_random_rerank"] = d1

    d2 = {"median_sd_ratio": float(np.median(diag["sd_ratio"])), "min_sd_ratio": float(np.min(diag["sd_ratio"])),
          "mean_overlap_bm25_top100": float(np.mean(diag["overlap_bm25_top100"])),
          "min_overlap_bm25_top100": float(np.min(diag["overlap_bm25_top100"])),
          "hybrid_unreranked_minus_bm25": paired(np.array(extra["hybrid_unreranked"]) - np.array(pq["bm25"]))}
    if d2["median_sd_ratio"] > 3 and d2["mean_overlap_bm25_top100"] >= 0.7:
        d2["verdict"] = "'60% weighted' contradicted: BM25 dominates the pool"
    elif d2["median_sd_ratio"] < 1.5:
        d2["verdict"] = "'60% weighted' supported"
    else:
        d2["verdict"] = "mixed"
    R["D2_pool_composition"] = d2

    G = R["means"]["abl_topic_off"] - R["means"]["full_clinical"]
    A = R["means"]["abl_topic_off"] - R["means"]["topic_off_random_ties"]
    d3 = {"G_topic_off_minus_full": G, "A_stable_minus_random_ties": A,
          "A_over_G": A / G if G else None,
          "median_distinct_topic_off_scores_per_pool": float(np.median(diag["distinct_topic_off_scores"])),
          "max_distinct_topic_off_scores_per_pool": int(np.max(diag["distinct_topic_off_scores"])),
          "topic_off_random_ties_minus_full": paired(np.array(extra["topic_off_random_ties"]) - full)}
    d3["verdict"] = ("artefact" if G > 0 and A >= 0.5 * G else
                     "not an artefact" if G > 0 and A < 0.25 * G else
                     "partial" if G > 0 else "no topic-off gain to explain")
    R["D3_topic_off_ties"] = d3

    d4a = paired(np.array(extra["cos_only_rerank"]) - np.array(extra["random_rerank"]))
    d4a["verdict"] = ("harmful" if d4a["ci95"][1] < 0 else "informative" if d4a["ci95"][0] > 0
                      else "no detectable information")
    seed_means = rand_emb.mean(1)
    d4b = {"pubmedbert_full": float(full.mean()), "random_seed_means": seed_means.tolist(),
           "random_mean_of_seeds": float(seed_means.mean()),
           "mean_minus_pubmedbert": float(seed_means.mean() - full.mean()),
           "max_abs_seed_minus_pubmedbert": float(np.max(np.abs(seed_means - full.mean()))),
           "paired_random_mean_minus_pubmedbert": paired(rand_emb.mean(0) - full)}
    d4b["verdict"] = ("insensitive to encoder" if abs(d4b["mean_minus_pubmedbert"]) <= MARGIN
                      and d4b["max_abs_seed_minus_pubmedbert"] <= 0.04 else "sensitive to encoder")
    R["D4a_cos_only_vs_random"] = d4a
    R["D4b_random_embeddings"] = d4b

    d5 = {k: float(np.median(diag[k])) for k in ["rho_we", "rho_cos", "rho_action"]}
    d5["queries_with_constant_factor"] = rho_nan
    d5["verdict"] = ("evidence weight dominates" if d5["rho_we"] > d5["rho_cos"] + 0.2
                     else "evidence weight does not dominate")
    R["D5_score_drivers"] = d5

    R["D6_controls"] = {"bm25_rerank_of_pool_minus_bm25": paired(np.array(extra["bm25_rerank_of_pool"]) - np.array(pq["bm25"])),
                        "oracle_ceiling": R["means"]["oracle_rerank_of_pool"],
                        "mean_pool_recall_judged": float(np.mean(diag["pool_recall_judged"])),
                        "mean_bm25_top10_in_pool": float(np.mean(diag["bm25_top10_in_pool"]))}
    R["D6_controls"]["positive_control_ok"] = abs(R["D6_controls"]["bm25_rerank_of_pool_minus_bm25"]["mean"]) <= 0.01

    abl = ["abl_evidence_off", "abl_topic_off", "abl_apply_off", "abl_action_off"]
    tests = {a: paired(np.array(pq[a]) - full) for a in abl}
    for a, p in zip(abl, holm([tests[a]["p_signflip"] for a in abl])):
        tests[a]["p_holm"] = p
        tests[a]["detectable"] = p < 0.05
    tests["abl_temporal_off_identical"] = bool(np.array_equal(pq["abl_temporal_off"], full))
    R["D7_ablation_paired_tests"] = tests

    R["D8_precision_at_10"] = {"bm25": R["means"]["p10_bm25"], "full": R["means"]["p10_full"],
                               "default_json_threshold": 0.75}
    R["runtime_seconds"] = round(time.time() - t0, 1)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(R, f, indent=2)
    cols = configs + list(extra) + [f"random_embedding_seed{s}" for s in range(N_EMB_SEEDS)]
    with open(os.path.join(args.out, "per_query.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid"] + cols)
        for qi, (qid, _) in enumerate(queries):
            row = [pq[c][qi] for c in configs] + [extra[c][qi] for c in extra] + [rand_emb[s][qi] for s in range(N_EMB_SEEDS)]
            w.writerow([qid] + [f"{v:.10f}" for v in row])
    print(json.dumps({k: R[k] for k in R if k.startswith("D")}, indent=2))
    print(f"wrote {args.out}/results.json and per_query.csv")


if __name__ == "__main__":
    main()

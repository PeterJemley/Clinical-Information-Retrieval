"""
Pre-registered evaluation runner for the Clinical IR framework.

Runs BM25 baseline, the full clinical relevance pipeline, and ablations on
BEIR NFCorpus (test split), scoring every configuration with the repository's
OWN metric (clinical_ir.evaluate.ndcg_at_k) so baseline and system are
measured identically.

Design decisions (all disclosed in evals/report.md):
  * Topical component needs real embeddings; we use the exact encoder named in
    config.json (PubMedBERT), mean-pooled per config ("pooling": "mean").
  * NFCorpus ships no publication dates -> temporal is NEUTRALISED by setting
    every doc's publication_date equal to query_time (tau == 1.0 for all docs).
    We decline to impute or scrape dates. Temporal is therefore reported as
    UNEVALUATED, not as a passed/failed ablation.
  * NFCorpus ships no study-type metadata -> evidence_level comes from the
    framework's own keyword classifier (a disclosed PROXY). We run the
    evidence-off ablation mechanically but report criterion 3 as UNEVALUATED.
  * No population metadata -> applicability_relevance() returns its 0.5 default
    for every doc (a per-query additive constant -> inert for ranking).

Nothing in the framework's parameters (alpha, lambda, evidence table, beta) is
altered. Ablations only zero a factor to measure its contribution.
"""
import json, os, sys, hashlib
from datetime import datetime
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, ".cache")
DATA = os.path.join(ROOT, "datasets", "nfcorpus")
os.makedirs(CACHE, exist_ok=True)

from clinical_ir.relevance import (
    Document, Query, clinical_relevance,
    topical_relevance, applicability_relevance, actionability_relevance,
    COMPONENT_WEIGHTS,
)
from clinical_ir.temporal import temporal_decay
from clinical_ir.evidence import evidence_weight
from clinical_ir.retrieval import BM25, hybrid_search, clinical_rerank
from clinical_ir.evaluate import ndcg_at_k  # repo's own metric

MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
K = 10
CAND_K = 100  # candidate pool the framework reranks (retrieval.py default usage)


# ---------- data loading ----------
def load_corpus():
    docs = []
    with open(os.path.join(DATA, "corpus.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            docs.append((r["_id"], r["title"], r["text"]))
    return docs


def load_queries_and_qrels():
    qrels = {}  # qid -> {docid: grade}
    with open(os.path.join(DATA, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = float(score)
    qtext = {}
    with open(os.path.join(DATA, "queries.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            qtext[r["_id"]] = r["text"]
    queries = [(qid, qtext[qid]) for qid in qrels if qid in qtext]
    queries.sort()
    return queries, qrels


# ---------- embeddings (PubMedBERT, mean pooled) ----------
def embed_texts(texts, tag):
    key = hashlib.md5((MODEL + "|" + tag + "|" + str(len(texts))).encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"emb_{tag}_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    import torch
    from transformers import AutoTokenizer, AutoModel
    os.environ.setdefault("HF_HOME", os.path.join(CACHE, "hf"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL); mdl.eval()
    out = []
    bs = 32
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            hs = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pooling
        out.append(vecs.cpu().numpy().astype(np.float32))
        if (i // bs) % 20 == 0:
            print(f"  [{tag}] embedded {i+len(batch)}/{len(texts)}", flush=True)
    arr = np.vstack(out)
    np.save(path, arr)
    return arr


# ---------- scoring (faithful re-implementation of R(d,q) with per-factor toggles) ----------
def score_doc(doc, query, weights, use_tau=True, use_we=True, query_time=None):
    tau = temporal_decay(doc.publication_date, query_time, doc.domain) if use_tau else 1.0
    we = evidence_weight(doc.evidence_level) if use_we else 1.0
    comp = (weights["topic"] * topical_relevance(doc, query)
            + weights["apply"] * applicability_relevance(doc, query)
            + weights["action"] * actionability_relevance(doc, query))
    return tau * we * comp


def rerank(cands, docs, query, weights, use_tau=True, use_we=True, query_time=None):
    scored = [(idx, score_doc(docs[idx], query, weights, use_tau, use_we, query_time))
              for idx, _ in cands]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def w_without(component):
    w = dict(COMPONENT_WEIGHTS)
    w[component] = 0.0
    return w


# ---------- metric with bootstrap CI ----------
def bootstrap_ci(per_query, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    arr = np.asarray(per_query)
    means = [arr[rng.randint(0, len(arr), len(arr))].mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    print("Loading NFCorpus...", flush=True)
    corpus = load_corpus()
    queries, qrels = load_queries_and_qrels()
    print(f"  {len(corpus)} docs, {len(queries)} test queries", flush=True)

    doc_ids = [d[0] for d in corpus]
    doc_texts = [f"{t} {x}".strip() for _, t, x in corpus]
    q_texts = [t for _, t in queries]

    print("Embedding documents (PubMedBERT, mean-pooled)...", flush=True)
    doc_emb = embed_texts(doc_texts, "docs")
    print("Embedding queries...", flush=True)
    q_emb = embed_texts(q_texts, "queries")

    # Fixed reference time; publication_date == query_time so tau == 1.0 for all docs.
    ref_time = datetime(2020, 1, 1)
    documents = [
        Document(id=doc_ids[i], title=corpus[i][1], abstract=corpus[i][2],
                 content_embedding=doc_emb[i], publication_date=ref_time,
                 population=None)  # evidence_level auto-classified (keyword proxy)
        for i in range(len(corpus))
    ]

    bm25 = BM25()
    bm25.index(documents)

    # sanity: our score_doc (no ablation) must equal the shipped clinical_relevance
    q0 = Query(text=queries[0][1], embedding=q_emb[0], population=None)
    a = score_doc(documents[0], q0, COMPONENT_WEIGHTS, query_time=ref_time)
    b = clinical_relevance(documents[0], q0, query_time=ref_time)
    assert abs(a - b) < 1e-9, f"scorer mismatch {a} vs {b}"
    print(f"Scorer fidelity check OK (score_doc == clinical_relevance: {a:.6f})", flush=True)

    configs = ["bm25", "full_clinical", "abl_temporal_off", "abl_evidence_off",
               "abl_topic_off", "abl_apply_off", "abl_action_off"]
    per_query = {c: [] for c in configs}

    for qi, (qid, qtext) in enumerate(queries):
        query = Query(text=qtext, embedding=q_emb[qi], population=None)
        rel_map = qrels[qid]  # graded gains {docid: grade}

        # BM25 baseline (ranks all docs)
        bm25_ids = [documents[i].id for i, _ in bm25.search(qtext, CAND_K)]
        per_query["bm25"].append(ndcg_at_k(bm25_ids, rel_map, K))

        # candidate pool the framework reranks
        cands = hybrid_search(query, documents, bm25, k=CAND_K)

        def ndcg_for(ranked):
            ids = [documents[idx].id for idx, _ in ranked]
            return ndcg_at_k(ids, rel_map, K)

        per_query["full_clinical"].append(
            ndcg_for(rerank(cands, documents, query, COMPONENT_WEIGHTS, True, True, ref_time)))
        per_query["abl_temporal_off"].append(
            ndcg_for(rerank(cands, documents, query, COMPONENT_WEIGHTS, False, True, ref_time)))
        per_query["abl_evidence_off"].append(
            ndcg_for(rerank(cands, documents, query, COMPONENT_WEIGHTS, True, False, ref_time)))
        per_query["abl_topic_off"].append(
            ndcg_for(rerank(cands, documents, query, w_without("topic"), True, True, ref_time)))
        per_query["abl_apply_off"].append(
            ndcg_for(rerank(cands, documents, query, w_without("apply"), True, True, ref_time)))
        per_query["abl_action_off"].append(
            ndcg_for(rerank(cands, documents, query, w_without("action"), True, True, ref_time)))

        if qi % 50 == 0:
            print(f"  query {qi+1}/{len(queries)}", flush=True)

    results = {}
    for c in configs:
        pq = per_query[c]
        mean = float(np.mean(pq))
        lo, hi = bootstrap_ci(pq)
        results[c] = {"ndcg@10": mean, "ci95": [lo, hi], "n": len(pq)}

    base = results["bm25"]["ndcg@10"]
    full = results["full_clinical"]["ndcg@10"]
    summary = {
        "dataset": "BEIR/NFCorpus test",
        "n_docs": len(corpus),
        "n_queries": len(queries),
        "metric": "NDCG@10 (repo clinical_ir.evaluate.ndcg_at_k, graded gains)",
        "encoder": MODEL,
        "bm25_ndcg@10": base,
        "full_ndcg@10": full,
        "delta_abs": full - base,
        "delta_rel_pct": (full - base) / base * 100 if base > 0 else None,
        "criterion_5pct_abs_met": (full - base) >= 0.05,
        "configs": results,
    }
    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== RESULTS ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

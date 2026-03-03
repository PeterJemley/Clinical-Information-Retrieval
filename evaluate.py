"""
Evaluation module - tests the bold conjectures.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .relevance import Document, Query
from .retrieval import BM25, hybrid_search, clinical_rerank


MINIMUM_IMPROVEMENT = 0.05


@dataclass
class EvalResult:
    method: str
    p_at_10: float
    ndcg_at_10: float
    mrr: float


def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k if k > 0 else 0.0


def dcg(scores: List[float], k: int) -> float:
    return sum(s / np.log2(i + 2) for i, s in enumerate(scores[:k]))


def ndcg_at_k(retrieved: List[str], rel_map: Dict[str, float], k: int) -> float:
    scores = [rel_map.get(d, 0) for d in retrieved[:k]]
    ideal = sorted(rel_map.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal, k)
    return dcg(scores, k) / ideal_dcg if ideal_dcg > 0 else 0.0


def mrr(retrieved: List[str], relevant: set) -> float:
    for i, d in enumerate(retrieved):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compare_methods(docs, queries, judgments, bm25):
    agg = {"bm25": [], "hybrid": [], "clinical": []}
    
    for query, relevant in zip(queries, judgments):
        rel_map = {d: 1.0 for d in relevant}
        
        bm25_ids = [docs[i].id for i, _ in bm25.search(query.text, 100)]
        agg["bm25"].append((precision_at_k(bm25_ids, relevant, 10),
                           ndcg_at_k(bm25_ids, rel_map, 10),
                           mrr(bm25_ids, relevant)))
        
        hybrid_ids = [docs[i].id for i, _ in hybrid_search(query, docs, bm25, k=100)]
        agg["hybrid"].append((precision_at_k(hybrid_ids, relevant, 10),
                             ndcg_at_k(hybrid_ids, rel_map, 10),
                             mrr(hybrid_ids, relevant)))
        
        cands = hybrid_search(query, docs, bm25, k=100)
        clin_ids = [docs[i].id for i, _ in clinical_rerank(cands, docs, query, 100)]
        agg["clinical"].append((precision_at_k(clin_ids, relevant, 10),
                               ndcg_at_k(clin_ids, rel_map, 10),
                               mrr(clin_ids, relevant)))
    
    return {m: EvalResult(m, np.mean([x[0] for x in v]),
                          np.mean([x[1] for x in v]),
                          np.mean([x[2] for x in v]))
            for m, v in agg.items()}


def test_conjecture(baseline: float, experimental: float) -> Tuple[bool, str]:
    imp = experimental - baseline
    if imp >= MINIMUM_IMPROVEMENT:
        return True, f"PASS: +{imp:.3f}"
    elif imp > 0:
        return False, f"FAIL: +{imp:.3f} < {MINIMUM_IMPROVEMENT} required"
    else:
        return False, f"FAIL: no improvement ({imp:.3f})"

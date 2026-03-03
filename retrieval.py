"""
Tier 1: Evidence Aggregation - Hybrid Search

Combines BM25 (keyword) with semantic search.
score_hybrid = beta * normalize(BM25) + (1-beta) * cos(E_d, E_q)

Type B parameter: beta = 0.4 (subject to sensitivity analysis)
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
import math
import re

from .relevance import Document, Query, clinical_relevance, cosine_similarity


HYBRID_BETA = 0.4


class BM25:
    """BM25 baseline - what we test against."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = Counter()
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.corpus_size = 0
        self.doc_term_freqs = []
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def index(self, documents: List[Document]):
        self.corpus_size = len(documents)
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.doc_freqs = Counter()
        
        for doc in documents:
            tokens = self._tokenize(f"{doc.title} {doc.abstract}")
            self.doc_lengths.append(len(tokens))
            term_freqs = Counter(tokens)
            self.doc_term_freqs.append(term_freqs)
            for term in set(tokens):
                self.doc_freqs[term] += 1
        
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
    
    def score(self, query: str, doc_idx: int) -> float:
        query_terms = self._tokenize(query)
        doc_tf = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0.0
        for term in query_terms:
            if term not in doc_tf:
                continue
            tf = doc_tf[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length))
            score += idf * tf_norm
        return score
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def z_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    if std == 0:
        return [0.0] * len(scores)
    return [(s - mean) / std for s in scores]


def hybrid_search(query: Query, documents: List[Document], bm25: BM25, 
                  beta: float = HYBRID_BETA, k: int = 100) -> List[Tuple[int, float]]:
    bm25_scores = [bm25.score(query.text, i) for i in range(len(documents))]
    bm25_norm = z_normalize(bm25_scores)
    semantic = [cosine_similarity(doc.content_embedding, query.embedding) for doc in documents]
    hybrid = [beta * bm25_norm[i] + (1 - beta) * semantic[i] for i in range(len(documents))]
    return sorted(enumerate(hybrid), key=lambda x: x[1], reverse=True)[:k]


def clinical_rerank(candidates: List[Tuple[int, float]], documents: List[Document],
                    query: Query, k: int = 10) -> List[Tuple[int, float]]:
    scored = [(idx, clinical_relevance(documents[idx], query)) for idx, _ in candidates]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


if __name__ == "__main__":
    from datetime import datetime
    np.random.seed(42)
    
    docs = [
        Document(id=f"doc_{i}", title=f"Document {i} about transfusion",
                 abstract=f"Abstract {i} on blood management",
                 content_embedding=np.random.randn(768),
                 publication_date=datetime(2015 + i % 10, 1, 1),
                 population={"age_min": 40, "age_max": 80})
        for i in range(50)
    ]
    
    docs[0] = Document(
        id="best_doc", 
        title="Systematic review of transfusion thresholds cardiac surgery",
        abstract="Meta-analysis recommends 70 g/L threshold guideline",
        content_embedding=np.random.randn(768),
        publication_date=datetime(2023, 1, 1),
        population={"age_min": 60, "age_max": 85}
    )
    
    query = Query(text="transfusion threshold cardiac surgery",
                  embedding=np.random.randn(768),
                  population={"age_min": 65, "age_max": 85})
    
    bm25 = BM25()
    bm25.index(docs)
    
    print("Retrieval Module Test")
    print("=" * 40)
    print("\nBM25 top 3:")
    for idx, score in bm25.search(query.text, 3):
        print(f"  {docs[idx].id}: {score:.2f}")
    
    print(f"\nHybrid (beta={HYBRID_BETA}) top 3:")
    for idx, score in hybrid_search(query, docs, bm25, k=3):
        print(f"  {docs[idx].id}: {score:.2f}")
    
    print("\nClinical rerank top 3:")
    candidates = hybrid_search(query, docs, bm25, k=10)
    for idx, score in clinical_rerank(candidates, docs, query, k=3):
        print(f"  {docs[idx].id}: {score:.3f}")

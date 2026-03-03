"""
Clinical relevance function: R(d, q) = τ(d,t) · w_e(d) · Σ αᵢ · rᵢ(d,q)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from .temporal import temporal_decay
from .evidence import EvidenceLevel, evidence_weight, classify_evidence_level


@dataclass
class Document:
    id: str
    title: str
    abstract: str
    content_embedding: np.ndarray
    publication_date: datetime
    evidence_level: Optional[EvidenceLevel] = None
    population: Optional[Dict] = None
    domain: str = "pbm"
    
    def __post_init__(self):
        if self.evidence_level is None:
            self.evidence_level = classify_evidence_level(self.title, self.abstract)


@dataclass 
class Query:
    text: str
    embedding: np.ndarray
    population: Optional[Dict] = None


# Type B parameters
COMPONENT_WEIGHTS = {"topic": 0.50, "apply": 0.30, "action": 0.20}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def topical_relevance(doc: Document, query: Query) -> float:
    return cosine_similarity(doc.content_embedding, query.embedding)


def applicability_relevance(doc: Document, query: Query) -> float:
    doc_pop = doc.population or {}
    query_pop = query.population or {}
    if not doc_pop or not query_pop:
        return 0.5
    
    overlap_min = max(doc_pop.get("age_min", 0), query_pop.get("age_min", 0))
    overlap_max = min(doc_pop.get("age_max", 120), query_pop.get("age_max", 120))
    query_range = query_pop.get("age_max", 120) - query_pop.get("age_min", 0)
    
    if overlap_max <= overlap_min or query_range == 0:
        return 0.0
    return (overlap_max - overlap_min) / query_range


def actionability_relevance(doc: Document, query: Query) -> float:
    text = f"{doc.title} {doc.abstract}".lower()
    signals = ["recommend", "guideline", "threshold", "should be", "dose", "protocol"]
    return min(sum(1 for s in signals if s in text) / len(signals), 1.0)


def clinical_relevance(doc: Document, query: Query, 
                       query_time: Optional[datetime] = None,
                       weights: Optional[Dict[str, float]] = None) -> float:
    if query_time is None:
        query_time = datetime.now()
    if weights is None:
        weights = COMPONENT_WEIGHTS
    
    tau = temporal_decay(doc.publication_date, query_time, doc.domain)
    w_e = evidence_weight(doc.evidence_level)
    
    component_score = (
        weights["topic"] * topical_relevance(doc, query) +
        weights["apply"] * applicability_relevance(doc, query) +
        weights["action"] * actionability_relevance(doc, query)
    )
    
    return tau * w_e * component_score


if __name__ == "__main__":
    np.random.seed(42)
    
    doc = Document(
        id="pmid_12345",
        title="Randomized trial of restrictive transfusion thresholds",
        abstract="RCT comparing 70 g/L vs 100 g/L threshold, recommends restrictive approach",
        content_embedding=np.random.randn(768),
        publication_date=datetime(2020, 1, 1),
        population={"age_min": 50, "age_max": 80},
    )
    
    query = Query(
        text="transfusion threshold elderly cardiac surgery",
        embedding=np.random.randn(768),
        population={"age_min": 65, "age_max": 85},
    )
    
    print("Clinical Relevance Module - Self Test")
    print("=" * 50)
    print(f"Document: {doc.title}")
    print(f"Evidence: {doc.evidence_level.name} (w={evidence_weight(doc.evidence_level):.2f})")
    print(f"Temporal decay: {temporal_decay(doc.publication_date):.3f}")
    print(f"Topical: {topical_relevance(doc, query):.3f}")
    print(f"Applicability: {applicability_relevance(doc, query):.3f}")
    print(f"Actionability: {actionability_relevance(doc, query):.3f}")
    print(f"\nFINAL SCORE: {clinical_relevance(doc, query):.3f}")

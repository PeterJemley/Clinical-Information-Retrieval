"""
Clinical Information Retrieval Framework

A falsifiable framework for domain-specific information retrieval
that treats relevance as multidimensional.
"""

__version__ = "0.1.0"

from .temporal import temporal_decay, DECAY_RATES
from .evidence import EvidenceLevel, evidence_weight, classify_evidence_level
from .relevance import Document, Query, clinical_relevance
from .retrieval import BM25, hybrid_search, clinical_rerank

__all__ = [
    "temporal_decay",
    "DECAY_RATES",
    "EvidenceLevel",
    "evidence_weight",
    "classify_evidence_level",
    "Document",
    "Query",
    "clinical_relevance",
    "BM25",
    "hybrid_search",
    "clinical_rerank",
]

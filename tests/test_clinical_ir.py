"""Tests for clinical_ir package."""

import numpy as np
from datetime import datetime
import pytest

from clinical_ir import (
    temporal_decay, DECAY_RATES, EvidenceLevel, evidence_weight,
    Document, Query, clinical_relevance, BM25, hybrid_search, clinical_rerank,
)


class TestTemporalDecay:
    def test_current_document_full_relevance(self):
        now = datetime.now()
        assert temporal_decay(now, now, "pbm") == pytest.approx(1.0)
    
    def test_decay_is_monotonic(self):
        now = datetime.now()
        tau_new = temporal_decay(datetime(2024, 1, 1), now, "pbm")
        tau_old = temporal_decay(datetime(2014, 1, 1), now, "pbm")
        assert tau_new > tau_old
    
    def test_domain_rates_differ(self):
        now = datetime.now()
        old = datetime(2016, 1, 1)
        assert temporal_decay(old, now, "infectious_disease") < temporal_decay(old, now, "anatomy")


class TestEvidenceHierarchy:
    def test_systematic_review_highest(self):
        assert evidence_weight(EvidenceLevel.SYSTEMATIC_REVIEW) == 1.0
    
    def test_hierarchy_order(self):
        w_sr = evidence_weight(EvidenceLevel.SYSTEMATIC_REVIEW)
        w_rct = evidence_weight(EvidenceLevel.RCT)
        w_exp = evidence_weight(EvidenceLevel.EXPERT_OPINION)
        assert w_sr > w_rct > w_exp


class TestRelevance:
    def test_relevance_bounded(self):
        np.random.seed(42)
        doc = Document(
            id="test", title="RCT transfusion", abstract="Study",
            content_embedding=np.random.randn(768),
            publication_date=datetime(2020, 1, 1),
        )
        query = Query(text="transfusion", embedding=np.random.randn(768))
        score = clinical_relevance(doc, query)
        assert 0 <= score <= 1


class TestRetrieval:
    def test_bm25_returns_results(self):
        np.random.seed(42)
        docs = [
            Document(id=f"doc_{i}", title=f"Doc {i}", abstract=f"Text {i}",
                     content_embedding=np.random.randn(768),
                     publication_date=datetime(2020, 1, 1))
            for i in range(20)
        ]
        query = Query(text="test", embedding=np.random.randn(768))
        bm25 = BM25()
        bm25.index(docs)
        assert len(bm25.search(query.text, k=10)) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

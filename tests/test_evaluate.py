import pytest
from types import SimpleNamespace
from clinical_ir import evaluate
import math

# Simple local stubs for Document and Query if needed (we'll use SimpleNamespace)
class Doc(SimpleNamespace):
    pass

class Query(SimpleNamespace):
    pass

def test_precision_at_k_basic():
    retrieved = ["d1", "d2"]
    relevant = {"d2"}
    # k=2, denom = min(2,2) => 2, hits=1 -> 0.5
    assert evaluate.precision_at_k(retrieved, relevant, 2) == pytest.approx(0.5)

def test_precision_at_k_no_results():
    retrieved = []
    relevant = {"d1"}
    assert evaluate.precision_at_k(retrieved, relevant, 10) == 0.0
    assert evaluate.precision_at_k(retrieved, relevant, 0) == 0.0

def test_dcg_ndcg_simple():
    rel_map = {"d1": 2.0, "d2": 1.0}
    retrieved = ["d2", "d1"]  # less ideal ordering
    ndcg = evaluate.ndcg_at_k(retrieved, rel_map, 2)
    # compute expected via the same formula for sanity (since floating)
    scores = [rel_map.get(d, 0.0) for d in retrieved[:2]]
    ideal = sorted(rel_map.values(), reverse=True)[:2]
    expected = evaluate.dcg(scores, 2) / evaluate.dcg(ideal, 2)
    assert ndcg == pytest.approx(expected)

def test_mrr_basic():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"c"}
    # c at index 2 -> reciprocal rank 1/3
    assert evaluate.mrr(retrieved, relevant) == pytest.approx(1/3)

def test_compare_methods_integration_stubs():
    # Docs and query
    docs = [Doc(id="d0"), Doc(id="d1"), Doc(id="d2")]
    queries = [Query(text="foo")]
    judgments = [{"d1"}]  # d1 is relevant

    # BM25 stub: yields indices (0,1,2)
    class BM25Stub:
        def search(self, text, k):
            return [(0, 1.0), (1, 0.5), (2, 0.2)]

    bm25 = BM25Stub()

    # Hybrid stub: returns (1,2,0) so d1 at top
    def hybrid_stub(query, docs_arg, bm25_arg, k=100):
        return [(1, 0.9), (2, 0.8), (0, 0.5)]

    # Clinical re-ranker stub: reorder hybrid candidates to move relevant to position 1
    def clinical_stub(cands, docs_arg, query, k=100):
        # pretend re-ranker pushes d2 first then d1 then d0
        return [(2, 1.0), (1, 0.9), (0, 0.1)]

    # Monkeypatch the module-level functions used by compare_methods
    evaluate.hybrid_search = hybrid_stub
    evaluate.clinical_rerank = clinical_stub

    results = evaluate.compare_methods(docs, queries, judgments, bm25)
    # Assertions based on the stubs:
    # BM25 ranking: ["d0","d1","d2"] -> relevant d1 at index 1 -> mrr 1/2
    assert results["bm25"].mrr == pytest.approx(0.5)
    # Hybrid ranking: ["d1","d2","d0"] -> relevant at index 0 -> mrr 1.0
    assert results["hybrid"].mrr == pytest.approx(1.0)
    # Clinical ranking: ["d2","d1","d0"] -> relevant at index 1 -> mrr 0.5
    assert results["clinical"].mrr == pytest.approx(0.5)

def test_conjecture_boundaries():
    base = 0.10
    # Exactly MINIMUM_IMPROVEMENT should pass
    ok, msg = evaluate.test_conjecture(base, base + evaluate.MINIMUM_IMPROVEMENT)
    assert ok is True
    assert "PASS" in msg
    # Slight improvement but below threshold should fail
    ok2, msg2 = evaluate.test_conjecture(base, base + evaluate.MINIMUM_IMPROVEMENT / 2)
    assert ok2 is False
    assert "FAIL" in msg2
    # No improvement
    ok3, msg3 = evaluate.test_conjecture(base, base - 0.01)
    assert ok3 is False
    assert "no improvement" in msg3
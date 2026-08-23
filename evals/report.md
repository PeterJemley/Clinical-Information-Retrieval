# Pre-registered evaluation: results

**Date:** 2026-08-23
**Dataset:** BEIR / **NFCorpus**, test split
**Corpus size:** 3,633 documents
**Queries:** 323 test queries (all with ≥1 judgment)
**Relevance judgments:** NFCorpus official `qrels/test.tsv` — graded human relevance
(grades 1 and 2; 12,334 judgments), distributed with the BEIR benchmark. Source:
Boteva et al., *A Full-Text Learning to Rank Dataset for Medical Information
Retrieval* (ECIR 2016), as repackaged by BEIR
(`public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip`).
**Metric:** NDCG@10, computed with the repository's own
`clinical_ir.evaluate.ndcg_at_k` (linear gains) so baseline and system are scored
identically. Confidence intervals are 2,000-sample bootstrap over the 323 queries.
**Encoder (topical component):** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`,
mean-pooled — the exact model and pooling named in `config.json`.
**Harness:** `evals/run_eval.py`. Raw numbers: `evals/results.json`.

---

## Headline result

| Configuration | NDCG@10 | 95% CI | vs BM25 |
|---|---|---|---|
| **BM25 baseline** | **0.310** | [0.277, 0.345] | — |
| **Full clinical system** | **0.071** | [0.056, 0.085] | **−0.240 (−77.3%)** |

The registered bar was **≥ +5% NDCG@10 over BM25**. The full multidimensional
pipeline came in **77% *below* BM25**, with non-overlapping confidence intervals.
The 323-query sample amply supports intervals, and they are decisive here.

**By the framework's own falsification criterion — "equal to counts as failure" —
the core conjecture is refuted on this dataset. It did not merely fail to clear
+5%; it performed far worse than the lexical baseline it set out to beat.**

The BM25 baseline (0.310) lands close to BEIR's published NFCorpus BM25 NDCG@10
(~0.325), which is evidence the metric and preprocessing are calibrated correctly
and the gap is real, not an artifact of scoring.

---

## Ablations

All ablation rows rerank the **same** hybrid top-100 candidate pool as the full
system, changing exactly one factor. "Contribution" = full − ablation (positive =
the component *helps*; negative = it *hurts*).

| Configuration | NDCG@10 | 95% CI | Contribution of removed factor |
|---|---|---|---|
| Full clinical system | 0.0705 | [0.0557, 0.0849] | — |
| Temporal decay **off** (τ≡1) | 0.0705 | [0.0557, 0.0849] | **0.000** (identical) |
| Evidence weight **off** (w_e≡1) | 0.0670 | [0.0530, 0.0820] | +0.0035 |
| Topical score **off** (α_topic=0) | 0.0824 | [0.0671, 0.0979] | **−0.0119** (removing it *helps*) |
| Applicability **off** (α_apply=0) | 0.0703 | [0.0555, 0.0847] | +0.0002 (negligible) |
| Actionability **off** (α_action=0) | 0.0731 | [0.0581, 0.0879] | −0.0026 |

Two things stand out:

1. **Turning the temporal factor off changes nothing** — the rows are bit-identical.
   That is expected: NFCorpus ships no dates, so τ was neutralised to 1.0 (see below).
   There is no temporal signal to ablate.
2. **Turning the topical component off *improves* the clinical system** (0.0705 →
   0.0824). The mean-pooled PubMedBERT cosine signal is actively harmful inside this
   pipeline. But note: even the *best* ablation (0.0824) is still ~3.8× below BM25
   (0.310). **No configuration of the multidimensional machinery comes near the
   lexical baseline.**

---

## The three registered criteria

### 1. ≥5% NDCG@10 improvement over BM25 — **FAILED**

Full system 0.071 vs BM25 0.310 → **−77.3%**, non-overlapping 95% CIs. Not met,
and not close. This is the primary registered criterion and it is refuted.

### 2. Temporal-decay contribution demonstrated separately — **COULD NOT BE EVALUATED**

NFCorpus documents carry only a PubMed `url` in their metadata (verified across all
3,633 docs) — **no publication dates**. The framework's `temporal_decay` requires a
publication date, and `Document` has no default for it. Rather than impute dates or
scrape PubMed (an external substitution the task forbids), every document's date was
set equal to query time, giving τ ≡ 1.0 for all documents. The temporal factor is
therefore **neutralised, not tested** — which is why "temporal off" is identical to
the full system. This criterion remains **untested, not passed**. Evaluating it would
require a corpus with reliable, per-document publication dates.

### 3. Evidence-weight contribution demonstrated separately — **COULD NOT BE EVALUATED**

NFCorpus ships **no study-type metadata** (RCT / observational / review /
meta-analysis). Evidence levels were therefore assigned by the framework's own
keyword classifier (`classify_evidence_level`) — **a proxy, disclosed here as such.**
Two independent reasons make this not a valid test of the conjecture:

- The evidence labels are inferred from title/abstract keywords, not ground-truth
  study design, so any effect measures the keyword heuristic, not evidence hierarchy.
- NFCorpus relevance judgments reward **topical** relevance to nutrition/medical
  questions. Nothing in the gold labels rewards ranking stronger study designs higher,
  so even a perfect evidence signal has no ground truth to improve against.

The ablation was run mechanically for completeness (evidence-off 0.0670 vs on 0.0705;
a +0.0035 difference with heavily overlapping CIs), but this **does not** demonstrate
the evidence-weight conjecture and is **not** counted toward it. This criterion remains
**untested, not passed**. Evaluating it would require a corpus with gold study-type
labels *and* judgments sensitive to evidence quality.

---

## Components that could not be evaluated, and why (summary)

| Component | Status | Reason |
|---|---|---|
| Topical (`r_topic`) | Evaluated | Real PubMedBERT embeddings generated (specified encoder) |
| Applicability (`r_apply`) | Effectively inert | No population metadata → constant 0.5 for every doc → no ranking effect |
| Temporal (`τ`) | Unevaluated | No publication dates in corpus; τ neutralised to 1.0 |
| Evidence (`w_e`) | Unevaluated (valid test) | No gold study-type; keyword proxy; judgments don't reward evidence quality |

---

## Observations (not recommendations, and explicitly not post-hoc tuning)

The following are reported as mechanism, **not** as changes to make the bar. Per the
task's binding constraints, no parameter (α weights, decay λ, evidence table, hybrid
β) was altered after seeing any result, and none should be.

- The dominant driver of the collapse is the **topical/semantic path**. `config.json`
  specifies mean-pooled PubMedBERT, and mean-pooled BERT embeddings are known to be
  poorly calibrated for cosine similarity without similarity fine-tuning. The full
  pipeline reranks a hybrid pool that is itself 60% weighted toward this signal, then
  reranks again by a score dominated by the same signal. The topic-off ablation
  improving on topic-on is direct evidence the signal is net-harmful here. This is a
  property of the framework's *specified* configuration, evaluated faithfully — not a
  substitution on my part.
- Because three of the four distinctive dimensions were inert or untestable on
  NFCorpus (applicability constant, temporal neutralised, evidence a proxy), this
  evaluation primarily tested the **topical reranking** path. A genuine test of the
  full multidimensional thesis needs a corpus with publication dates, gold study-type
  labels, and judgments sensitive to recency/evidence — which BEIR medical subsets do
  not provide.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" torch transformers
# NFCorpus into datasets/ (gitignored); then:
python evals/run_eval.py   # writes evals/results.json
```

## Repository defects noted in passing (not modified)

- The shipped test suite does not run under `pytest`: the package uses relative
  imports (`from .relevance import ...`) with an `__init__.py` at the repo root, and
  `tests/test_evaluate.py` does a bare `import evaluate`. Neither test module collects.
- `config.json` names PubMedBERT as the encoder, but no code in the repository loads
  any embedding model; the module self-tests use `np.random.randn` vectors. The
  embedding step in `evals/run_eval.py` supplies this missing piece using the specified
  model.

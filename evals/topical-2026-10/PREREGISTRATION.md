# Preregistration: the topical component alone, with a similarity-trained encoder

Date: 2026-10-04    Base commit: `6f88985`    Status: confirmatory for the primary comparison
and the predictions below; everything else is labelled secondary or exploratory

This file is committed before any computation for this test is run. No MedCPT embedding has
been computed and no number below comes from this test. I have seen the August results
(`evals/results.json`) and the 3 October audit (`evals/audit-2026-10-03/`).

## The question

The README registers this as the next test: the topical component alone, with a
similarity-trained encoder in place of the August mean-pooled PubMedBERT, measured against
BM25 on the same data. If it still loses, the semantic path is the problem. If it wins, the
August instrument fault is isolated and the composite becomes worth testing.

The audit sharpened the question. Inside the August candidate pool the cosine signal was
indistinguishable from a random order (D4a), so it was uninformative rather than harmful, and
the composite passes a strong topical signal through (E2, exploratory). The question here is
whether a similarity-trained encoder carries real topical signal on this corpus.

The README's phrase "in place of mean pooling" is imprecise. The variable under test is
similarity training. Pooling follows from the encoder chosen.

This is a diagnostic of the instrument. It is not a re-test of criterion 1, which concerns the
full system and failed on 23 August. That verdict is not revisited.

## Fixed before running

- **Data.** BEIR/NFCorpus from the HuggingFace mirror, the same revisions and checksums the
  audit recorded (`BeIR/nfcorpus` rev `b5026a0e`, `BeIR/nfcorpus-qrels` rev `a451b3b2`). Test
  split: 3,633 documents, 323 queries, 12,334 graded judgments.
- **Encoder (primary, the only one).** MedCPT. Queries through `ncbi/MedCPT-Query-Encoder` rev
  `d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc`, max length 64. Documents through
  `ncbi/MedCPT-Article-Encoder` rev `d05a736da4bb84ee4057b7f7999485be6ed85465`, input
  `[title, text]` as a sentence pair, max length 512. Both use the `[CLS]` last hidden state,
  as the model card specifies. CPU, float32.
- **Topical score.** `clinical_ir.relevance.topical_relevance`, cosine similarity, called as
  the framework defines it. MedCPT was trained with a dot product. Cosine is primary because it
  is the framework's topical component; dot product is reported as exploratory.
- **Baseline.** `clinical_ir.retrieval.BM25` with its defaults (k1 = 1.5, b = 0.75), unchanged,
  scored exactly as August. It must reproduce 0.3103 within 0.001, or the harness is broken and
  nothing below is reported as a result.
- **Metric.** `clinical_ir.evaluate.ndcg_at_k`, k = 10, graded gains.
- **No framework file and no parameter in `default.json` is changed.** The encoder is supplied
  by this test's harness, as the August harness supplied its own.

## Primary comparison

For each query, rank all 3,633 documents by cosine between the MedCPT query and article
vectors. Score NDCG@10. Compare with BM25 per query.

Statistic: per-query paired difference (topical − BM25) over the 323 queries; mean, 95%
paired-bootstrap CI (10,000 resamples, seed 0), two-sided sign-flip permutation p (10,000
draws, seed 0). Same procedure as the audit.

| Outcome | Condition on the CI of (topical − BM25) |
|---|---|
| Equivalent | CI entirely inside [−0.03, +0.03] |
| Topical beats BM25 | CI lower bound > 0 and not "equivalent" |
| Topical loses to BM25 | CI upper bound < 0 and not "equivalent" |
| Inconclusive | anything else |

Reading the outcome:
- **Beats:** the August topical fault was the instrument. The composite becomes worth testing,
  under its own preregistration.
- **Loses:** the semantic path is the problem even with a similarity-trained encoder.
- **Equivalent or inconclusive:** the topical path carries signal comparable to BM25's but gives
  no reason to prefer it. Reported as such, not as a win.

The framework's own "≥5%" bar is reported beside the outcome under both readings (absolute
+0.05 and relative +5%), since the audit found the two are not reconciled. It does not decide
this test.

## Prediction

MedCPT's paper reports zero-shot NFCorpus results. A search summary gives NDCG@10 = 0.355 for
MedCPT against 0.325 for BM25. I could not read the paper's text from this environment, so I
have not confirmed whether 0.355 is the retriever alone or retriever plus re-ranker, and I do
not rely on it. This repository's BM25 scores 0.310, not 0.325.

Prediction: topical NDCG@10 between 0.30 and 0.37, with a point guess of 0.33. Paired
difference about +0.02. Under the rule above I expect **inconclusive or beats**, and I think
"beats" is slightly more likely than not. The prediction is refuted if the outcome is "loses".

## Secondary checks (reported, not decisive)

- **S1. Pool rerank vs random.** Rerank BM25's top 100 by MedCPT cosine and compare with a
  random reordering of the same pool (200 draws per query, `numpy.default_rng([query_index, 7])`,
  as in the audit). Only queries with at least 100 non-zero BM25 scores, because `BM25.search`
  pads short lists with zero-score documents (audit E1). This repeats the audit's D4a with the
  new encoder. Informative if the CI lower bound > 0.
- **S2. The August encoder, run the same way.** Full-corpus cosine ranking with mean-pooled
  PubMedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` rev `e1354b7a`,
  the audit's resolution of the August model name). Reports MedCPT − PubMedBERT directly.
- **S3. Composite with MedCPT.** The full R(d, q), parameters as specified, reranking the BM25
  top-100 pool with MedCPT as the topical vector, against BM25. This is a look ahead to the
  composite test, not that test.

## Controls

- **Positive control.** BM25 reordering of its own top 100 must equal BM25 within 0.001.
- **Negative control.** Random query and document vectors (seeds 0 to 4), full-corpus cosine
  ranking. Expected near zero. If any seed scores within 0.05 of MedCPT, the encoder is not
  what the primary result measures.
- **Ties.** The number of tied scores in each ranked top 10 is counted for every configuration.
  If any configuration has a median of fewer than 50 distinct scores per 100-document pool,
  it is also scored with random tie-breaking (200 draws), and both values are reported.
- **Oracle composite.** R(d, q) with the gold grade as the topical score on the BM25 pool, so
  the composite's capacity to pass a strong signal is checked on the pool S3 uses.

## Exploratory

- MedCPT dot product instead of cosine.
- Per-query breakdown by number of BM25 matches (fewer than 10, 10 to 99, 100 or more).

## Records

New directory `evals/topical-2026-10/`: this file, `run_topical.py`, `manifest.json` (data and
model revisions, checksums, package versions, seeds, script sha256), `results.json`,
`per_query.csv`, `run.log`, `REPORT.md`. Embedding caches are keyed on model, revision and a
hash of the input texts. Nothing in the 23 August record (`evals/report.md`,
`evals/results.json`, `evals/run_eval.py`, `evals/run.log`,
`evals/what-the-failure-establishes.md`) or in `evals/audit-2026-10-03/` is edited.

## Limitations known in advance

- One dataset, the same one as August and the audit. Not chosen after seeing a result for this
  encoder, but not a test of generality either.
- MedCPT was trained on PubMed search logs, and NFCorpus documents are PubMed articles. It was
  not trained on NFCorpus queries or judgments, as far as its documentation says. Exposure to
  the documents' text during training cannot be ruled out.
- The ±0.03 equivalence margin is carried over from the audit and set before this run.

## Compute and stopping rule

One run on CPU. If a bug in `run_topical.py` is found after results are seen, it is fixed and
logged below with the reason, and affected results are labelled exploratory. No second encoder
is tried in this test, whatever the outcome.

## Deviation log

- 2026-10-04: the first run wrote `results.json`, `per_query.csv` and the console summary, then
  stopped before writing `manifest.json`, because `scipy` (imported only to record its version)
  was not installed. Installed `scipy` and reran the unchanged script with the embedding cache
  from the first run. `results.json` and `per_query.csv` were byte-identical to the first run's.
  Seen results before the change: yes. Effect: none on results.
- 2026-10-04: added `exploratory.py` after seeing S3, to find why the composite collapses with
  MedCPT. Seen results before the change: yes. Effect: exploratory.
- 2026-10-04: the S2 PubMedBERT embeddings are not byte-identical to the audit's, because this
  harness batches inputs by length. S2 was never registered as a reproduction. Effect: none on
  the decision rule.

# Preregistration: audit of the 23 August NFCorpus evaluation

Date: 2026-10-03    Base commit: `3b0dd21`    Status: confirmatory for the predictions below; anything else is exploratory

This file is committed before any audit computation is run. At the time of writing I have
seen the August aggregate numbers (`evals/results.json`) and nothing else: no per-query
values, no re-run, and no output of the checks below.

## What is being audited

The August run reported NDCG@10 = 0.0705 for the full clinical system against 0.3103 for
BM25 on BEIR/NFCorpus (323 test queries), and the write-ups (`evals/report.md`,
`evals/what-the-failure-establishes.md`, README) explain the loss mainly by the topical
signal: mean-pooled PubMedBERT cosine is "actively harmful", the candidate pool is "60%
weighted toward this signal", and the topic-off ablation (0.0824) is "direct evidence the
signal is net-harmful". The verdict on criterion 1 is not in question here: it failed by a
wide margin. The question is whether the stated mechanism is supported, and whether the
test could tell the encoder apart from the rest of the pipeline.

## Claims under test (mine, about the August pipeline)

1. Within its own candidate pool, the clinical reranker performs no better than a random
   reordering of that pool. The loss to BM25 comes from discarding the pool's order, not
   from the topical signal being harmful.
2. The hybrid candidate pool is dominated by BM25, not 60% by the embedding signal,
   because `z(BM25)` is spread over several standard deviations while cosine similarities
   of mean-pooled BERT vectors sit in a narrow band.
3. The topic-off "improvement" is mostly a tie-breaking artefact: with α_topic = 0 the
   score takes a few dozen distinct values, and Python's stable sort keeps the hybrid
   (≈ BM25) order inside each tie.
4. The August result is insensitive to the encoder: replacing PubMedBERT vectors with
   random vectors leaves the full-system NDCG@10 about where it is.

## Competing hypotheses

| Hypothesis | Predicts |
|---|---|
| H1, the write-ups' account: the embedding signal is harmful and drives the loss | cos-only rerank below random; random embeddings score clearly better than PubMedBERT; topic-off gain survives random tie-breaking |
| H2, my account: the reranker carries ~no information; the loss is the discarded pool order | full ≈ random rerank; random embeddings ≈ PubMedBERT; topic-off gain disappears under random tie-breaking |
| Bug in the August harness or a code/data mismatch | re-run fails to reproduce the August numbers |
| Bug in this audit's script | its re-scoring fails to reproduce its own harness numbers exactly (fidelity check) |

## Design

- Data: BEIR/NFCorpus from the HuggingFace mirror (`BeIR/nfcorpus` rev `b5026a0e`,
  `BeIR/nfcorpus-qrels` rev `a451b3b2`), because the UKP host the August run used is
  blocked from this environment. Checksums go in the run manifest. Pre-run checks already
  done: 3,633 docs, 323 test queries, 12,334 judgments (11,758 grade 1, 576 grade 2),
  matching `evals/report.md`.
- Encoder: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` (now redirected
  to `BiomedBERT`, rev `e1354b7a`), mean-pooled, CPU, float32.
- Code under test: the repository at `3b0dd21`, made importable as `clinical_ir` through a
  symlink, because `pip install -e .` installs no `clinical_ir` package (checked).
- Step R, reproduction: run a byte-identical copy of `evals/run_eval.py`.
- Step D, diagnostics: `evals/audit-2026-10-03/audit.py`, which first re-derives every
  August configuration and must match step R's means to 1e-9 before any diagnostic is
  trusted. It saves per-query values for every configuration.
- Randomness: random reranks and random tie-breaks use 200 seeds per query, averaged to an
  expected NDCG per query. Random-embedding controls use 5 seeds (0–4).
- Statistic for every paired comparison: per-query paired difference over the 323 queries;
  mean, 95% paired-bootstrap CI (10,000 resamples, seed 0), and a two-sided sign-flip
  permutation p-value (10,000 draws, seed 0). Holm correction across the four ablation
  tests in D7.
- Planning note on power: if per-query paired differences have SD ≈ 0.15, the SE over 323
  queries is ≈ 0.008 and the smallest effect detectable with 80% power is ≈ 0.023. That is
  why the equivalence margin below is ±0.03 and not tighter. The August topic-off gap
  (0.012) is below this, so even a significant paired test there would be fragile.

## Checks and decision rules

**R. Reproduction.** BM25 within 0.001 of 0.3103 (deterministic). Full system and each
ablation within 0.003 of the August values (CPU vs Apple numerics may move a few ties).
Fidelity print within 1e-3 of 0.551196. If any fails: report non-reproduction, and every
diagnostic below describes this run, not August's.

**D1, primary. Full clinical rerank vs random rerank of the same pool.**
| Outcome | Condition on the CI of (full − random) |
|---|---|
| At chance | CI entirely inside [−0.03, +0.03] |
| Better than chance | CI lower bound > 0 and not "at chance" |
| Worse than chance | CI upper bound < 0 and not "at chance" |
| Inconclusive | anything else |
Prediction: at chance. Claim 1 is refuted if the reranker is better than chance.

**D2. Pool composition.** Per query: SD over the corpus of 0.4·z(BM25) and of 0.6·cos;
overlap of hybrid top-100 with BM25 top-100; NDCG@10 of the hybrid order unreranked.
Prediction: median SD ratio > 5, mean overlap ≥ 0.8, hybrid NDCG@10 within 0.03 of BM25.
The "60% weighted" claim is contradicted if median ratio > 3 and mean overlap ≥ 0.7,
supported if median ratio < 1.5, mixed otherwise.

**D3. Topic-off tie-breaking.** Count distinct topic-off scores per pool. Re-score topic-off
with random tie-breaking. Let G = topic-off(stable) − full and A = topic-off(stable) −
topic-off(random ties). Artefact if A ≥ 0.5·G; not an artefact if A < 0.25·G; partial
otherwise. Prediction: artefact.

**D4. Is the embedding signal informative, and is the result sensitive to it?**
(a) cos-only rerank of the pool vs random rerank: harmful if CI upper < 0, informative if
CI lower > 0, no detectable information otherwise. No confident prediction; weakly
informative is my guess.
(b) Full pipeline (pool + rerank) with random embeddings, seeds 0–4. Insensitive to the
encoder if the 5-seed mean is within ±0.03 of the PubMedBERT full system and every seed
within ±0.04; sensitive otherwise. Prediction: insensitive. Claim 4 is refuted if sensitive.

**D5. What drives the clinical score inside the pool.** Per query, Spearman ρ of the final
score with each of w_e, cos and actionability; median over queries. "Evidence weight
dominates" if median ρ(w_e) > median ρ(cos) + 0.2. Prediction: evidence weight dominates.

**D6. Controls that bound the test.** Positive control: rerank the pool by BM25 score;
should be within 0.01 of BM25 NDCG@10, or the harness is broken. Ceiling: rerank the pool
by gold grade (oracle). Reports how much room a good reranker had. Pool recall of judged
documents reported descriptively.

**D7. Paired tests for the August ablations** (evidence-off, topic-off, apply-off,
action-off vs full; temporal-off is identical by construction). Holm-adjusted p < 0.05
counts as a detectable difference. No prediction.

**D8, secondary.** P@10 for BM25 and the full system, against the 0.75 threshold that
`default.json` registers, because that file names a different primary outcome
(`decision_concordance`) from the one evaluated.

## Compute and stopping rule

One reproduction run and one diagnostic run on CPU. No parameter of the framework is
changed. If a bug in `audit.py` is found after results are seen, it is fixed, logged below
as a deviation with the reason, and the affected results are labelled exploratory.

## Deviation log

- (none yet)

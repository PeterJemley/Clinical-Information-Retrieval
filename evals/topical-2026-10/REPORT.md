# The topical component alone, with a similarity-trained encoder

**Date:** 2026-10-04
**Preregistration:** [`PREREGISTRATION.md`](PREREGISTRATION.md), committed in `a20b856` before any
computation for this test. Deviations are logged at its foot.
**Code under test:** the framework at `6f88985`, unchanged. `topical_relevance`, `BM25`,
`clinical_relevance` and `ndcg_at_k` are imported from `clinical_ir`. The harness supplies only
the encoders.
**Run record:** [`manifest.json`](manifest.json) (data checksums, model revisions, embedding
hashes, package versions, seeds). Per-query values: [`per_query.csv`](per_query.csv).

The 23 August record and `evals/audit-2026-10-03/` are not edited.

---

## Verdict

**Inconclusive, by the preregistered rule.** Ranking the whole corpus by MedCPT cosine scores
NDCG@10 = **0.3303** against BM25's **0.3103**. The paired difference is +0.0200, 95% CI
[−0.0019, +0.0422], p = 0.070. The interval crosses zero, so this is not a win. It reaches
above +0.03, so it is not "equivalent" either. The design could detect an effect of about 0.031
with 80% power. The observed +0.020 is below that.

**The prediction held.** I predicted 0.30 to 0.37 with a point guess of 0.33, a difference of
about +0.02, and an outcome of "inconclusive or beats". The prediction would have been refuted
by "loses". It did not lose.

**What this settles.** The README framed two outcomes. If the topical component still lost to
BM25, the semantic path was the problem. If it won, the instrument fault was isolated. The
topical path did not lose. With a similarity-trained encoder it ranks about as well as BM25,
and slightly better on the point estimate. The August encoder, run the same way, scores 0.0327
(S2). On this corpus the semantic path works when the encoder is trained for similarity. The
August topical failure was the instrument.

**What it does not settle.** "The composite becomes worth testing" does not follow. Put MedCPT
into the full R(d, q) as specified and the composite scores **0.0883** (S3), about where August
was, against 0.3303 for MedCPT alone on the full corpus and 0.3239 for MedCPT reranking the same
pool. With a working encoder, the composite still destroys the order. The exploratory follow-up
shows why: within a pool, MedCPT cosine moves the score by a tenth of what the keyword evidence
weight does.

The framework's "≥5%" bar, reported as preregistered and not deciding this test: +6.4%
relative, so the relative reading is met; +0.020 absolute against +0.05, so the absolute reading
is not.

---

## 1. Primary comparison

| Configuration | NDCG@10 |
|---|---|
| BM25 (repository `BM25`, as August) | 0.3103 |
| **MedCPT cosine, full corpus** | **0.3303** |

| Paired difference (MedCPT − BM25) | Value |
|---|---|
| Mean | +0.0200 |
| 95% paired-bootstrap CI | [−0.0019, +0.0422] |
| Sign-flip p | 0.070 |
| Queries favouring MedCPT / BM25 / tied | 115 / 105 / 103 |
| Detectable effect at 80% power | 0.031 |
| Preregistered verdict | **inconclusive** |

## 2. Controls

| Control | Requirement | Outcome |
|---|---|---|
| BM25 reproduces August | within 0.001 of 0.3103 | 0.3103001667, exact. **Pass** |
| BM25 reordering of its own top 100 | within 0.001 of BM25 | 0.3103. **Pass** |
| Random query and document vectors, seeds 0–4 | near zero; none within 0.05 of MedCPT | 0.0088 to 0.0122. **Pass** |
| Distinct scores per 100-document pool (median) | random tie-breaking if fewer than 50 | BM25 95, MedCPT cosine 99, composite with MedCPT 99, oracle composite 12. Only the oracle composite triggered the rule. Random ties give 0.4459, the same as the stable sort. |
| Ties at rank 10 on the full corpus | counted | BM25 82 queries (mostly short match lists padded with zero scores), MedCPT cosine 5, PubMedBERT 3 |
| Oracle composite on the BM25 pool | passes a strong topical signal | 0.4459 against a ceiling of 0.5330 |

The data matched the audit's checksums exactly (`corpus.jsonl`, `queries.jsonl`,
`qrels/test.tsv`), and the counts matched the preregistration (3,633 / 323 / 12,334).

## 3. Secondary checks

| Check | Result | Reading |
|---|---|---|
| S1. MedCPT cosine reranking BM25's top 100, against a random reorder of the same pool (196 queries with ≥ 100 BM25 matches) | 0.3131 vs 0.0628; +0.2503 [+0.2140, +0.2863] | Informative. The audit's D4a found August's cosine indistinguishable from random here (+0.0084). |
| S2. August encoder (mean-pooled PubMedBERT), full corpus, same way | 0.0327. MedCPT − PubMedBERT = +0.2976 [+0.2656, +0.3301] | The encoder change is what moved the topical path, by about 0.30. |
| S3. Composite R(d, q), parameters as specified, MedCPT as the topical vector, reranking BM25's top 100 | 0.0883. Composite − BM25 = −0.2220 [−0.2529, −0.1916]. Composite − MedCPT alone = −0.2420 [−0.2721, −0.2122] | A working topical signal does not rescue the composite. |

## 4. Exploratory

These were not decisive by design, or were added after the results were seen.

**Dot product instead of cosine** (registered as exploratory). MedCPT was trained with a dot
product. Scored that way it reaches 0.3655: +0.0552 over BM25, CI [+0.0331, +0.0781], and
+0.0352 over its own cosine, CI [+0.0239, +0.0470]. This clears both readings of the "≥5%" bar.
But it is not what the framework's topical component computes, so it does not count toward the
verdict. Using it would be a change to `topical_relevance`, and a test of that change would need
its own registration.

**By number of BM25 matches** (registered as exploratory):

| Queries | n | BM25 | MedCPT cosine | Difference, 95% CI |
|---|---|---|---|---|
| Fewer than 10 BM25 matches | 79 | 0.2528 | 0.2812 | +0.0284 [−0.0162, +0.0764] |
| 10 to 99 | 48 | 0.5026 | 0.4787 | −0.0239 [−0.0751, +0.0256] |
| 100 or more | 196 | 0.2864 | 0.3138 | +0.0274 [−0.0006, +0.0552] |

No stratum is decisive on its own.

**Why the composite collapses** (`exploratory.py`, written after seeing S3). Over BM25's top
100 for each query, medians across queries:

| Quantity | Median |
|---|---|
| SD of 0.5 · cosine within the pool | 0.024 |
| SD of 0.2 · actionability within the pool | 0.021 |
| SD of the evidence weight w_e within the pool | 0.188 |
| Cosine range within the pool | 0.41 to 0.67 |
| Spearman of the final score with w_e / cosine / actionability | 0.82 / 0.51 / 0.33 |

R(d, q) multiplies the sum of the components by w_e, which runs from 0.25 to 1.0 and is set by
the keyword evidence classifier. Within a pool the topical term varies by a few hundredths, so
the evidence class decides the order. The audit found the same with the August encoder
(Spearman 0.71 with w_e). It also found that the composite passes an *oracle* topical score
(E2), because that score spans 0 to 1. A real encoder's cosines sit in a narrow band. The
composite cannot use a topical signal at the scale a real encoder delivers it, however good
the ranking that signal carries.

## 5. What follows

1. The README's registered next step assumed a win here would make the composite worth testing.
   S3 shows the composite as specified fails with a working encoder. Before a composite test is
   worth registering, the scale mismatch between the topical term and w_e has to be addressed in
   the framework. Rescaling cosine within the pool or making w_e additive are both Type B design
   changes, and either one needs registering before it is tested.
2. The evidence classifier the audit found mis-firing ("rct" matching "infarction") is what
   decides the composite's order. Fixing it is a precondition for any composite test, on this
   corpus or another.
3. A decisive topical-versus-BM25 result on NFCorpus would need more queries than the 323 test
   queries provide, or a larger effect. The dot-product row suggests the effect may be larger
   under the encoder's own similarity function. That is a hypothesis for a registered test, not
   a finding.

## 6. Limitations

- One dataset, the same one as August and the audit.
- MedCPT was trained on PubMed search logs, and NFCorpus documents are PubMed articles. Exposure
  to the documents' text during training cannot be ruled out. It was not trained on NFCorpus
  queries or judgments, as far as its documentation says.
- The published MedCPT figure behind the prediction came from a search summary. I could not
  read the paper from this environment (arxiv.org, academic.oup.com and PMC are blocked).
- The PubMedBERT embeddings for S2 are not byte-identical to the audit's, because this harness
  batches by length. S2 is a fresh measurement, not a reproduction.
- The ±0.03 margin was carried over from the audit. The design's 80%-power detectable effect
  (0.031) is larger than the observed +0.020, so a real effect of this size would often come out
  inconclusive with 323 queries.

## Provenance

| Reported number | Source |
|---|---|
| 0.3103, 0.3303, 0.0327, 0.3655, random seeds | `results.json` `means` |
| +0.0200 [−0.0019, +0.0422], p 0.070, 115/105/103, 0.031 | `results.json` `primary_medcpt_cos_minus_bm25` |
| +6.4%, absolute bar 0.3603 | `results.json` `five_percent_bar` |
| control values, distinct scores, ties at rank 10 | `results.json` `controls` |
| S1, S2, S3 | `results.json` `secondary`; S1 pool means from `per_query.csv` restricted to `n_bm25_nonzero ≥ 100` |
| dot product, strata | `results.json` `exploratory` |
| SDs, cosine range, Spearman | `exploratory_results.json` |
| audit comparisons (D4a +0.0084, Spearman 0.71, E2) | `../audit-2026-10-03/REPORT.md` |

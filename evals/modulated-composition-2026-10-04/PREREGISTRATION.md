# Preregistration: query-modulated composition of the relevance modules

Date: 2026-10-04    Base commit: `b028a3b` (branch `claude/eloquent-gauss-r2otqh`, on top of `main` at `6f88985`)
Status: Stage 1 is confirmatory. Stage 0 is an exploratory pilot that checks the
plumbing and is never counted as evidence for the claim.

This plan is committed before any data for it has been touched. Some items, such as the
corpus, cannot be fixed until other work is done. They are listed under "Open items" and
must be fixed in a dated amendment to this file before any Stage 1 data are accessed. A
change made after Stage 1 data are accessed makes the affected results exploratory.

## Precondition

Stage 1 runs only after the topical-encoder test registered in the README ("What comes
next") has been run and reported. That test establishes whether the topical module carries
any signal. A recomposition can only exploit signal that the modules carry, so this test
comes second. Stage 1 uses whichever encoder that test used, whatever its outcome.

## Why this test

The 3 October audit (`evals/audit-2026-10-03/`) found that version 0.1 is rigid in three
specific ways.

1. **Hand-off without return.** BM25 builds the candidate pool, and then the reranker
   scores the pool without BM25. The pool in BM25 order scores NDCG@10 0.3102; reranked,
   it scores 0.0705 (`results.json`: `means.bm25_rerank_of_pool`, `means.full_clinical`).
2. **Influence set by score range, not by design.** With a weak topical signal, the
   0.25–1.0 evidence multiplier drives the ranking: median rank correlation 0.71, against
   0.25 for cosine (`D5_score_drivers`). With a perfect topical signal it does not:
   0.4993 against a ceiling of 0.5488 (`exploratory_results.json`, E2). The fixed form also
   multiplies the applicability constant by the evidence weight, so it is not inert
   (`D7`, apply-off).
3. **No dependence on the query.** Every module applies the same way to every question.
   The evidence weight depends only on the document. The decay rate comes from a document
   field that defaults to `"pbm"`.

The audit also found that no module other than BM25 carried detectable signal on NFCorpus
(`D1`, `D4a`, `D7`). The test below is built to tell those two explanations apart:
rigid architecture, or uninformative modules.

## Claim (Type C)

On held-out queries of a corpus that meets the eligibility criteria below, the modulated
composition M beats BM25 by at least 0.05 NDCG@10 (absolute), and also beats the fixed
version 0.1 composite. The gain is attributable to
the query-dependent gating of the modules, not to normalisation or to keeping BM25 in the
score.

## Competing hypotheses

| Hypothesis | Predicts for the primary outcome (M − BM25) and the controls |
|---|---|
| H1 modulation helps | M − BM25 ≥ 0.05; M beats U (same design, no gating); the noise control N does not reach the bar |
| H0 signal-limited | M ≈ BM25; M ≈ U ≈ N, because the modules carry no information |
| Keeping BM25 does the work | M − BM25 > 0, but N matches M |
| Flexibility, not design | M ≈ U, or a learned ranker L does as well as M |
| Insensitive test | the positive control P fails to beat M, or too few queries for the interval to settle |
| Bug | the fidelity checks fail (see Analysis) |

**My prior prediction:** refuted or inconclusive is more likely than supported, because the
audit found no informative module besides BM25. A supported result would be a substantive
surprise and would call for replication on a second corpus.

## Refuting result

The 95% interval for M − BM25 on held-out queries lies entirely below +0.05, while the
positive control passes.

## Configurations

All configurations except F rerank the same candidate pool: the BM25 top 100 from the
repository's `BM25` (k1 = 1.5, b = 0.75). NDCG@10 uses the repository's `ndcg_at_k` with
graded gains.

| Name | Definition |
|---|---|
| **B0** | BM25 order of the pool |
| **F** | Version 0.1 as specified: hybrid pool (β = 0.4), reranked by R(d, q). Run from tag `v0.1-august-eval` or commit `3b0dd21`. |
| **M** | Modulated composition, defined below |
| **U** | M with every gate set to 1 for every query. Same normalisation and same BM25 term, no modulation. |
| **L** | Learned ranker over the same features as M, plus question-type indicators and their products with each feature, with the same tuning budget (defined below) |
| **N** | Must-fail control. M with every non-BM25 module score replaced by standard-normal noise; seeds 0–4. |
| **R** | Mapping control. M with the gate-table columns assigned to question types by a random permutation; seeds 0–4. |
| **P** | Must-succeed control. M plus a synthetic module, gold grade + N(0, 1) noise, gated on for every query; seeds 0–4. |

### The modulated composition M

score_M(d, q) = Σᵢ gᵢ(c(q)) · zᵢ(d, q)

- zᵢ is module i's score standardised within the query's pool (mean 0, SD 1). A module with
  zero variance in a pool contributes 0. This makes each module's influence a matter of
  design rather than an accident of its score range.
- c(q) is the query's clinical question type, and gᵢ(c) ∈ {0, 1} comes from the fixed gate
  table below. Every module that is on gets the same weight. M has no fitted parameters.

Modules:
- **BM25:** the pool score. This keeps the lexical signal in the final score.
- **Topical:** cosine similarity, with the encoder from the precondition test.
- **Temporal:** τ from `temporal.py`, using the reference date fixed in the amendment.
- **Evidence:** w_e from `evidence.py`, applied to study-type labels from the corpus's
  metadata, never from the keyword classifier. A document with no label gets the pool
  mean, so z = 0.
- **Applicability:** from `relevance.py`, if the corpus has population metadata. If it
  does not, the module is gated off for every query, and this is recorded in the amendment.
- **Actionability:** the version 0.1 keyword module.

### Gate table (Type B, fixed here)

| Module | treatment | diagnosis / test | prognosis | other |
|---|---|---|---|---|
| BM25 | 1 | 1 | 1 | 1 |
| Topical | 1 | 1 | 1 | 1 |
| Evidence | 1 | 0 | 0 | 0 |
| Temporal | 1 | 1 | 1 | 0 |
| Applicability | 1 | 1 | 1 | 0 |
| Actionability | 1 | 1 | 0 | 0 |

Rationale:
- The version 0.1 evidence table ranks study designs as for treatment questions, with the
  RCT near the top. That ordering is wrong for diagnosis and prognosis questions, so the
  evidence module is off for them rather than applied with the wrong ordering.
- Currency and population fit matter for any clinical question, but not for background
  ("other") questions.
- Actionability (dosing, thresholds, recommendations) serves treatment and test questions,
  not prognosis.

Control R checks whether this particular mapping matters.

### Question type c(q)

If the corpus labels each query with a question type, those labels are used, mapped to
the four columns in the amendment. Otherwise, this rule is applied to the lower-cased
query text, case-insensitively and in this order. `\b` marks a word boundary.

- **treatment:** `\b(treat|therap|drug|dose|dosing|supplement|intervention|prevent)`
- **diagnosis / test:** `\b(diagnos|tests?\b|testing\b|screen|detect|marker|imaging)`
- **prognosis:** `\b(prognos|survival|risk of|outcome|mortality|recurrence)`
- **other:** everything else

The rule is fixed here and not tuned. `test` is matched only as a whole word, so that
words such as "testosterone" do not match, the kind of substring error the audit found
in the evidence classifier.

### The learned ranker L

L is a pairwise logistic ranker (linear RankNet), with L2 penalty chosen from
{0.01, 0.1, 1, 10} by 5-fold cross-validation on the training queries.

If the corpus has no training queries, L is cross-fitted over the test queries in 5
folds. Each fold is scored by a model trained on the other four, with the penalty chosen
by cross-validation nested inside those four. Fold assignment uses seed 0.

## Corpus eligibility (Stage 1)

Eligibility is checked from published documentation and file schemas only, never from
relevance judgments.

1. Publication dates for at least 95% of documents.
2. Study-type labels from metadata (for example, MEDLINE publication types), not inferred
   by this repository's keyword classifier.
3. Judgment guidelines under which currency or study design can affect relevance. If no
   available corpus meets this, the temporal and evidence parts of the claim are declared
   untestable in the amendment, as they were in August. The test then measures only
   whether gating the other modules helps.
4. A split in which no part of the design is set using test queries. M has no fitted
   parameters; this criterion matters for L.
5. At least 90 judged test queries. With fewer, a null result is declared inconclusive in
   advance unless the interval's upper end falls below +0.05.

Collections to check against these criteria first include the TREC Clinical Decision
Support and TREC Precision Medicine tracks. Whether either meets criterion 3 has not been
checked. The corpus must be chosen for meeting these criteria and fixed in the amendment
before Stage 1 data are touched. Choosing it this way is not selection by outcome: this
is a new conjecture, not a rerun of the August test.

## Sizing

The detectable effect depends on how far M departs from BM25. In the audit, the paired SD
against BM25 ranged from 0.011 (the hybrid ranking) to 0.280 (the full system). For
planning, take SD = 0.10. The smallest gain detectable with 80% power at a two-sided 5%
level is then:

| Test queries | Detectable gain (SD 0.10) | Detectable gain (SD 0.15) |
|---|---|---|
| 30 | 0.053 | 0.079 |
| 90 | 0.030 | 0.045 |
| 150 | 0.023 | 0.035 |
| 323 | 0.016 | 0.023 |

The 0.05 bar is detectable from about 90 queries. A relative 5% bar would need several
hundred queries, which is why the bar here is absolute.

## Analysis

- **Primary:** per-query paired differences, M − B0, on held-out test queries. Report the
  mean, a 95% paired-bootstrap interval (10,000 resamples, seed 0), and a two-sided
  sign-flip permutation p-value (10,000 draws, seed 0).
- **Secondary**, with Holm correction across the four comparisons of M: M − F, M − U,
  M − L, and M − R̄ (R̄ is the per-query mean over seeds).
- **Controls**, each judged by its 95% interval and not part of the Holm family: N̄ − B0
  and P̄ − M.
- **Fidelity checks before trusting any number:**
  - B0 equals BM25's own top-10 order.
  - F reproduces version 0.1's `clinical_relevance` ranking.
  - With every gate set to 0 except BM25's, M equals B0.
- Every configuration's per-query values are saved, along with a run manifest: code
  commit, data checksums, package versions, seeds and encoder revision.

## Decision rule

Apply the rows in this order. The first row whose condition holds is the verdict.

| Verdict | Condition |
|---|---|
| Test failed | P̄ − M interval lower bound ≤ 0. The harness cannot detect a real module, so no conclusion is drawn. |
| Refuted | M − B0 interval upper bound < 0.05 |
| Supported | M − B0 ≥ 0.05 with interval lower bound > 0, and M − F and M − U both > 0 (Holm-significant), and N̄ − B0 interval upper bound < 0.05 |
| Gain without attribution | M − B0 ≥ 0.05 with interval lower bound > 0, but M − U is not Holm-significant or N̄ − B0 reaches 0.05. The gain comes from normalisation or from keeping BM25, not from modulation. |
| Inconclusive | anything else |

Qualifiers, reported alongside the verdict without changing it:
- If L − M is Holm-significantly positive, a learned combination beats the designed one.
- If M − R̄ is not significant, the specific mapping from question type to modules adds
  nothing beyond gating as such.

## Stage 0: exploratory pilot on NFCorpus

This stage checks the plumbing and the controls before any Stage 1 data are touched. It
uses NFCorpus **dev** queries (`BeIR/nfcorpus-qrels`, `dev.tsv`). It does not use the test
queries, which the August evaluation and the audit have already used. L is trained on the
`train.tsv` queries. Question type comes from the rule above.

Predictions: |M − B0| < 0.03, |N̄ − B0| < 0.03, and P̄ − M > 0 with interval lower bound
> 0. If P̄ − M fails, the harness is fixed before Stage 1, and the fix is logged below.
Stage 0 results are reported as exploratory and are never used to change the gate table,
the modules or the decision rule.

## Compute and stopping rule

Each configuration runs once; N, R and P run once per seed (0–4). Nothing in M changes
after Stage 1 data are accessed. If anything must change, the change is logged below and
every affected result is labelled exploratory.

## Open items (to fix in a dated amendment before Stage 1 data are accessed)

- The corpus, its version and its checksums, and the result of each eligibility check
- The source of question types, and its mapping to the four gate columns
- The reference date for the temporal module
- The encoder, and the commit of the encoder test that used it
- The module code commit, and whether applicability can be computed
- The train / test split, if the corpus defines one

## Prior art

This applies existing ideas; it does not claim to introduce them. Using different ranking
models for different queries is query-dependent ranking (Geng et al., 2008). Scoring
citations against structured, evidence-based-medicine representations of the question is
Demner-Fushman and Lin (2007).

Demner-Fushman, D., & Lin, J. (2007). Answering clinical questions with knowledge-based
and statistical techniques. *Computational Linguistics*, 33(1), 63–103.
https://doi.org/10.1162/coli.2007.33.1.63

Geng, X., Liu, T.-Y., Qin, T., et al. (2008). Query dependent ranking using K-nearest
neighbor. *Proceedings of the 31st Annual International ACM SIGIR Conference*, 115–122.
https://doi.org/10.1145/1390334.1390356

## Amendments and deviation log

- (none yet)

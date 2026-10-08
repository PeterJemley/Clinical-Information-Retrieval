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

### Amendment 1 (committed 2026-10-08): the open items for Stage 1, and Stage 0

**Status: committed on 2026-10-08, before any Stage 0 or Stage 1 data for this test were
accessed.** It was drafted on 2026-10-04 and kept in a separate file until now (merged in
pull request #5, removed in this commit). It is committed before the A11 facts are
confirmed, so that the design is fixed before the Stage 0 pilot produces any numbers. If any
A11 fact turns out different, the difference is logged below as a deviation before Stage 1
data are accessed. A1–A11 were written from track documentation and published
descriptions. No TREC topic file, document or relevance judgment has been opened.

**A1. The precondition is met.** The topical-encoder test was preregistered in `a20b856`,
run in `0199399`, and merged in `7d752b1`. Its report is in `evals/topical-2026-10/`.
Under the rule above, Stage 1 uses its encoder, MedCPT:
- queries: `ncbi/MedCPT-Query-Encoder`, revision `d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc`,
  `[CLS]` pooling, maximum length 64;
- articles: `ncbi/MedCPT-Article-Encoder`, revision `d05a736da4bb84ee4057b7f7999485be6ed85465`,
  `[CLS]` pooling, maximum length 512, input `[title, abstract]` as a sentence pair.

This plan, including its prior prediction, was committed (`e6722f8`) before its author had
seen that test's result. The prediction is left unchanged.

**A2. A sixth eligibility criterion.** The topics must span at least two question types
whose gate-table columns differ. If every topic falls into one column, M and U are the same
system, so M − U is zero for every query and "Supported" cannot occur. This was found from
documentation, before any data were accessed.

It rules out the TREC Precision Medicine collections (2017–2020). Their topics are
structured as disease, gene, demographics and (in 2020) treatment, and carry no
question-type label. The registered rule would place every topic in one column.

PM 2020 is still worth noting for a different test. Its assessors judged strength of
evidence as a separate second phase, so it could test the framework's criterion 3 (evidence
weighting against uniform weights), which NFCorpus could not. That would be its own
preregistered test, not part of this one.

**A3. Corpus: TREC Clinical Decision Support 2014–2016, 90 topics.**

| Year | Topics | Collection each topic is judged against |
|---|---|---|
| 2014 | 30; diagnosis, test, treatment | PMC Open Access subset snapshot, 733,138 articles (NXML) |
| 2015 | 30, Task A only; same three types | the 2014 snapshot (to confirm, see A11) |
| 2016 | 30, from MIMIC-III admission notes; split evenly across the three types | PMC Open Access snapshot of 28 March 2016, 1.25 million articles |

Each topic is evaluated only against the collection its year's judgments were made on.
Per-topic paired differences are pooled across the 90 topics.

Eligibility against the six criteria:

| Criterion | Status |
|---|---|
| 1. Dates for ≥ 95% of documents | NXML front matter carries publication dates. The share is counted from document metadata before ranking anything. If it is below 95%, the temporal module is gated off for every query. |
| 2. Study types from metadata | MEDLINE publication types, reached through each article's PMID (see A5) |
| 3. Judgments that reward currency or study design | **Not established.** The judging guidelines could not be read from this environment (A11). If they do not say that currency or study design affects relevance, the temporal and evidence parts of the claim are declared untestable, as the plan already provides. |
| 4. No design choice set on test topics | No official training split. M has no fitted parameters. L is cross-fitted in 5 folds (seed 0), as already registered. |
| 5. At least 90 test topics | 90, exactly at the threshold |
| 6. At least two differing gate columns | Met: diagnosis and test topics use one column, treatment topics another |

**A4. Queries and question types.**
- Query text is the topic's **summary** field, the field all three years share. It is used
  for both BM25 and MedCPT; MedCPT truncates at 64 tokens.
- Question type is the topic's own type label: diagnosis and test map to the
  "diagnosis / test" column, treatment to "treatment". The registered regex rule is not
  used.
- 2016's longer note and description fields are not used.

**A5. Documents and modules.**
- **Document text:** the title and abstract from each article's NXML. If an article has no
  abstract, its title alone is used. Body text is not used, which keeps documents
  consistent with the framework's `Document(title, abstract)`.
- **Module code:** commit `7d752b1`. The framework files there are byte-identical to
  `3b0dd21` (`v0.1-august-eval`).
- **Temporal:** `temporal.py` with its default rate for `"pbm"` (λ = 0.12). The reference
  date is the snapshot date of the topic's collection: 28 March 2016 for 2016 topics, and
  the 2014 snapshot date for 2014 and 2015 topics (A11).
- **Evidence:** w_e from `evidence.py`, applied to MEDLINE publication types through this
  mapping, fixed here as a design choice (Type B):

  | MEDLINE publication type | Level |
  |---|---|
  | Systematic Review; Meta-Analysis | SYSTEMATIC_REVIEW |
  | Randomized Controlled Trial | RCT |
  | Controlled Clinical Trial; Clinical Trial (non-randomised); Observational Study; Comparative Study | CONTROLLED_OBSERVATIONAL |
  | Case Reports | UNCONTROLLED_OBSERVATIONAL |
  | Editorial; Comment; Letter; Consensus Development Conference | EXPERT_OPINION |

  When an article has several types, the highest level applies. An article with no PMID,
  or with none of these types, is unlabelled and gets the pool mean (z = 0). The keyword
  classifier is not used.
- **Applicability:** gated off for every query. NXML has no structured population metadata,
  as the plan's own rule requires.
- **Actionability:** the version 0.1 keyword module, applied to title and abstract.

**A6. BM25 at this scale.** The repository's pure-Python `BM25` cannot index about two
million documents in memory. Stage 1 uses a sparse-matrix version of the same formula: the
same tokenizer (`\b\w+\b`, lower-cased), k1 = 1.5, b = 0.75, and the same IDF expression.

Fidelity gate before any Stage 1 run: on the NFCorpus test queries, the new version must
reproduce the repository's top 100 for all 323 queries and NDCG@10 = 0.3103001667 to 1e-9.
If it does not, Stage 1 does not run.

**A7. F is redefined.** Version 0.1's R(d, q) now reranks the **BM25 top 100** with the
August encoder (mean-pooled `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`,
revision `e1354b7a`), embedding only the pooled documents. The original F used the hybrid
pool, which needs every document embedded; that is not feasible for about two million
articles here.

In the audit, the hybrid pool and BM25's top 100 overlapped by 0.975 on average for
queries where BM25 matched at least 100 documents (E1, exploratory). Whether that holds on these collections is
unknown, so F is reported as "version 0.1 reranking the BM25 pool", not as version 0.1 as
specified.

**A8. A new control, and a revised decision rule.** On these topics, M differs from U only
in the evidence module, which is switched off for diagnosis and test topics. A gain from
gating could therefore come from dropping that module, not from conditioning on question
type. A new configuration separates the two:

| Name | Definition |
|---|---|
| **V** | U with the evidence module switched off for every query |

- The Holm family becomes five comparisons: M − F, M − U, M − V, M − L, M − R̄.
- **Supported** now also requires M − V > 0 (Holm-significant).
- **Gain without attribution** now also applies when M − V is not Holm-significant.
- The other verdict rows are unchanged, and are still applied in the registered order.

**A9. Expected power.** M equals U on the 30 treatment topics, so M − U and M − V can be
non-zero on only 60 topics, which dilutes both comparisons. A non-significant result on
either is read as "no evidence that gating by question type helps", not as evidence that it
does not.

**A10. Where Stage 1 runs.** Not in this environment. Every host it needs is blocked here:
NIST TREC data, trec-cds.org, ir-datasets.com and NCBI. Stage 1 runs on a machine that can
reach them. It must record checksums for the topics, the qrels, both collection snapshots
and the PMID-to-publication-type table.

MedCPT and the August encoder embed only the pooled documents (at most 9,000), which is
feasible on CPU. The sparse BM25 index of about two million titles and abstracts needs
an estimated 16 GB of RAM.

**A11. Facts to confirm from documentation before any Stage 1 data access.** Each must be checked
in track documentation or topic-file schemas, never in judgments. A discrepancy is logged
here as a deviation before Stage 1 data are accessed.
- That 2015 used the 2014 collection.
- The 2014 snapshot date.
- That 2014 and 2015 each have 10 topics per type.
- What the 2014–2016 judging guidelines say about currency or study design (criterion 3).

**A12. How the plan applies to Stage 0 (NFCorpus, exploratory).** Fixed before any Stage 0
data are accessed.
- **Data:** the same HuggingFace mirror revisions as the audit. Stage 0 evaluates on the
  dev queries with at least one judgment in `dev.tsv`, and L trains on the training
  queries with at least one judgment in `train.tsv`. Test queries are used only for the A6
  BM25 fidelity gate. That gate runs BM25 alone, whose test score is already published.
- **Queries and question types:** the query is NFCorpus's query text. Question type comes
  from the registered regex rule, since NFCorpus has no type labels.
- **Pool:** the BM25 top 100 from the sparse BM25, used only after it passes the A6 gate.
- **Modules, as the registered rules require on this corpus:**
  - topical is MedCPT (A1);
  - temporal contributes 0, because NFCorpus has no dates;
  - evidence contributes 0, because NFCorpus has no study-type metadata, so every document
    is unlabelled. The A5 MEDLINE mapping is checked instead by unit tests on hand-written
    inputs.
  - applicability is off;
  - actionability is the version 0.1 keyword module;
  - BM25 is the pool score.
- **What that means for the comparisons:** M differs from U only by switching actionability
  off for prognosis and "other" queries. V equals U exactly, because evidence is inert here.
  That equality is checked, not treated as a finding.
- **F:** as in A7. The August encoder (mean-pooled PubMedBERT, revision `e1354b7a`) reranks
  the BM25 pool, with version 0.1's keyword evidence classifier and τ ≡ 1, as in August.
- **L:** a pairwise logistic ranker.
  - Features: the six module z-scores, plus each one multiplied by each of the four
    question-type indicators.
  - Pairs: up to 500 graded pairs per training query, drawn within its pool (seed 0).
  - Tuning: L2 penalty chosen from the registered grid by 5-fold cross-validation over
    training queries (seed 0), on mean NDCG@10, then refit on all training queries.
- **Seeds and ties:**
  - N, R and P use seeds 0–4.
  - R permutes the four gate columns across the four question types. A seed that draws the
    identity permutation is kept, not redrawn.
  - Ties are broken by the pool's BM25 order (a stable sort).
- **Embedding cache:** keyed by a hash of the texts, not by their count, which fixes the
  August cache-key flaw.
- **Outputs:** written to `stage0/`, with a run manifest.

Nothing Stage 0 shows changes M, the gate table, the modules or the decision rule.

**Sources for A2 and A3:**
- Roberts, K., Simpson, M. S., Demner-Fushman, D., et al. (2016). State-of-the-art in
  biomedical literature retrieval for clinical cases: A survey of the TREC 2014 CDS track.
  *Information Retrieval Journal*, 19(1–2), 113–148. https://doi.org/10.1007/s10791-015-9259-x
- Rybinski, M., Karimi, S., Nguyen, V., et al. (2020). A2A: A platform for research in
  biomedical literature search. *BMC Bioinformatics*, 21(S19).
  https://doi.org/10.1186/s12859-020-03894-8
- NIST TREC browser pages for CDS 2016 and PM 2020, and the TREC 2020 PM overview (Roberts
  et al., 2020), known here from web-search summaries only:
  https://pages.nist.gov/trec-browser/trec25/clinical/overview,
  https://pages.nist.gov/trec-browser/trec29/pm/proceedings

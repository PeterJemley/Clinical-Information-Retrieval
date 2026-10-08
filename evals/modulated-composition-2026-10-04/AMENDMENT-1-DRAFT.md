# Draft amendment 1 to `PREREGISTRATION.md`

## Amendment 1 (DRAFT, 2026-10-04): fixes the open items for Stage 1

**Status: draft, not part of the preregistration.** It binds only once it is moved into the
"Amendments and deviation log" section of `PREREGISTRATION.md` and committed there. That
must happen before any Stage 1 data are downloaded or opened. The items in A11 should be
confirmed first. Everything below was written from track documentation and published
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

**A11. Facts to confirm from documentation before any data access.** Each must be checked
in track documentation or topic-file schemas, never in judgments. A discrepancy is logged
here as a deviation before Stage 1 data are accessed.
- That 2015 used the 2014 collection.
- The 2014 snapshot date.
- That 2014 and 2015 each have 10 topics per type.
- What the 2014–2016 judging guidelines say about currency or study design (criterion 3).

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

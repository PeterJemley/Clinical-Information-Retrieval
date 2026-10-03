# Audit of the 23 August NFCorpus evaluation

**Date:** 2026-10-03
**Preregistration:** [`PREREGISTRATION.md`](PREREGISTRATION.md), committed in `a3117b4` before any
audit computation. Deviations are logged at its foot.
**Code under test:** repository at `3b0dd21`. Framework files are unchanged since `b27b17e`.
**Run record:** [`manifest.json`](manifest.json) (data checksums, encoder revision, package
versions, seeds, commands). Per-query values for every configuration: [`per_query.csv`](per_query.csv).

The August record (`evals/report.md`, `evals/results.json`, `evals/run_eval.py`) is not
edited. This directory sits beside it.

---

## Verdict

**The registered result stands.** A byte-identical rerun of `evals/run_eval.py` reproduces every
August number exactly, to the tenth decimal place, with identical confidence intervals. The
full clinical system scores NDCG@10 = 0.0705 against BM25's 0.3103, and criterion 1 fails under
every reading of "≥5%". The temporal and evidence criteria were correctly reported as untested.

**The explanation in the write-ups does not hold.** `evals/report.md` and
`evals/what-the-failure-establishes.md` attribute the loss mainly to the topical signal. They say
mean-pooled PubMedBERT cosine is "actively harmful", the candidate pool is "60% weighted toward
this signal", the final score is "dominated by the same signal", and the topic-off ablation is
"direct evidence the signal is net-harmful". The preregistered checks contradict each of these
claims:

| Write-up claim | What the audit found |
|---|---|
| Topic-off improving the system shows the topical signal is harmful | The improvement is a tie-breaking artefact. With α_topic = 0, each 100-document pool has a median of 9 distinct scores. Python's stable sort then keeps the BM25-like hybrid order within each tie. Break the ties at random and topic-off equals the full system exactly (difference +0.000007, 95% CI [−0.0044, 0.0044]). |
| The cosine signal is actively harmful | Inside the pool, ranking by cosine alone is indistinguishable from a random order (+0.0084, CI [−0.0034, 0.0206]). Replacing PubMedBERT with random vectors makes the pipeline *worse* (−0.0210, CI [−0.0295, −0.0134]), not better. The signal is uninformative, not harmful. |
| The final score is dominated by the same (topical) signal | Within the pool, the final score's median Spearman correlation is 0.71 with the evidence weight, 0.51 with actionability, and 0.25 with cosine. The keyword evidence weight drives the ranking. |
| The pool is 60% weighted toward the embedding signal | The hybrid ranking scores NDCG@10 = 0.3104, against 0.3103 for BM25, and is identical on 305 of 323 queries. The median spread of 0.4·z(BM25) is 73.7 times that of 0.6·cos. By the preregistered rule the verdict is "mixed", because my overlap prediction failed (see D2). |

**What produced the 0.24 loss.** The reranker takes a pool whose order is BM25's and replaces
that order with one indistinguishable from chance. Over the same pool, the full system scores
0.0705, a random reordering 0.0623, and BM25's own order 0.3102. The difference from random is
+0.0083, 95% CI [−0.0010, 0.0181], inside the preregistered ±0.03 band for "at chance". The
reranker's score is driven by a keyword evidence class and keyword actionability counts. Neither
tells you anything about NFCorpus's topical judgments.

**One exploratory result cuts the other way, against my own account.** Give the composite R(d, q)
a perfect topical score (the gold grade) and it still ranks the pool at 0.4993, against an
oracle ceiling of 0.5488 (E2). So the multiplicative evidence weight does not swamp a *strong*
topical signal. It swamps the weak one that mean-pooled PubMedBERT supplies. The README's
registered next test, a similarity-trained encoder for the topical component, therefore targets
a real deficit. It does so for a different reason than the write-ups give: the topical term is
uninformative, not harmful.

---

## 1. Reproduction (R)

Data came from the HuggingFace mirror of BEIR/NFCorpus, because the UKP host the August run used
is blocked here. Pre-run checks matched the August report: 3,633 documents, 323 test queries,
and 12,334 judgments. The August harness ran unmodified (sha256 recorded in the manifest), with
the repository imported as `clinical_ir` through a symlink. That symlink was needed because
`pip install -e .` installs no `clinical_ir` package.

| Configuration | August | Rerun | Difference |
|---|---|---|---|
| BM25 | 0.3103001667 | 0.3103001667 | 0 |
| Full clinical | 0.0705211375 | 0.0705211375 | 0 |
| Temporal off | 0.0705211375 | 0.0705211375 | 0 |
| Evidence off | 0.0669830409 | 0.0669830409 | 0 |
| Topic off | 0.0824347065 | 0.0824347065 | 0 |
| Applicability off | 0.0703160160 | 0.0703160160 | 0 |
| Actionability off | 0.0730688602 | 0.0730688602 | 0 |

The scorer-fidelity print also matches (0.551196). Every bootstrap interval is identical. The
audit script then re-derived all seven configurations from its own per-query loop and matched
the rerun to 1e-9 before it computed any diagnostic.

## 2. Preregistered checks: predictions and outcomes

Statistics are per-query paired differences over 323 queries, each with a 95% paired-bootstrap
CI and a two-sided sign-flip permutation p-value. Sources are in the provenance table at the end.

| Check | Prediction (committed first) | Outcome | Preregistered verdict |
|---|---|---|---|
| D1 full vs random rerank of the same pool | at chance | +0.0083 [−0.0010, 0.0181], p = 0.091 | **at chance** (within ±0.03). Prediction held. |
| D2 pool composition | SD ratio > 5; overlap ≥ 0.8; hybrid within 0.03 of BM25 | ratio 73.7; overlap 0.667; hybrid − BM25 = +0.00007 | **mixed**. The overlap prediction **failed**. |
| D3 topic-off ties | artefact | A/G = 0.999 (A = 0.0119 of G = 0.0119) | **artefact**. Prediction held. |
| D4a cosine-only vs random | none (guess: weakly informative) | +0.0084 [−0.0034, 0.0206], p = 0.179 | **no detectable information** |
| D4b random embeddings | insensitive | 5-seed mean 0.0496 vs 0.0705; worst seed 0.0233 away | **insensitive** by the margin. The paired test shows PubMedBERT beats noise: −0.0210 [−0.0295, −0.0134], p < 0.001. |
| D5 score drivers | evidence weight dominates | median ρ: w_e 0.71, action 0.51, cos 0.25 | **evidence weight dominates**. Prediction held. |
| D6 positive control | BM25 reorder of the pool within 0.01 of BM25 | −0.00006 | **passes**. Oracle ceiling 0.5488. |
| D7 August ablations (Holm) | none | only topic-off detectable (+0.0119, p_Holm = 0.0004) | topic-off is "significant", and D3 shows the cause is tie-breaking |
| D8 P@10 vs `default.json` threshold 0.75 | — | BM25 0.2180; full 0.0706 | the registered P@10 bar is out of reach even for BM25 on this corpus |

**D2, the failed prediction.** I predicted ≥ 0.8 overlap between the hybrid pool and BM25's top
100. The outcome was 0.667. Exploratory follow-up E1, run after seeing this, shows why. For 127
of the 323 queries, BM25 matches fewer than 100 documents, and 25 queries match none.
`BM25.search` then pads its top 100 with zero-score documents in corpus order, so overlap with
that list means little. For queries with at least 100 matches, overlap is 0.975. The measure,
not the claim, failed. Because the fault was found after seeing the outcome, the verdict stays
"mixed".

**D4b, read with care.** By the preregistered margin, the pipeline is insensitive to the encoder:
swapping PubMedBERT for noise moves NDCG@10 by 0.0210, against a 0.2398 gap. The paired test
nevertheless detects that difference, and it goes the opposite way from the write-ups' account.
Random vectors hurt, mostly by filling the pool's tail with random documents on the queries where
BM25 matches few. PubMedBERT contributes a little through pool membership and nothing detectable
through reranking (D4a).

**D1, every run.** Random reordering is averaged over 200 draws per query, so it is a smooth
expectation. The full system scores 0 on many queries and well on a few. That is why 169
queries favour random, 87 favour the full system, and 67 tie, while the mean difference is
slightly positive. The mean is the preregistered statistic. Per-query values are in
`per_query.csv`.

**The test could have passed.** Reordering the same pools by gold grade reaches 0.5488, far
above BM25 + 0.05. The pools held enough relevant documents for a good reranker to clear the
bar, and reordering them by BM25 score recovers BM25 exactly. The harness can detect both
directions. The system simply had no signal to give it.

## 3. Exploratory results (after seeing D1–D8)

| | Value |
|---|---|
| E1 median documents with non-zero BM25 per query | 405 |
| E1 queries with fewer than 100 BM25 matches / with none | 127 / 25 |
| E1 overlap with BM25 top 100, queries with ≥ 100 matches | 0.975 |
| E2 R(d,q) with oracle topical score, everything else as specified | 0.4993 |
| E2 the same with evidence weight off | 0.5488 (= ceiling) |

E2 is the "inject a known effect" check for the composite. A perfect topical signal survives the
evidence weighting with a loss of 0.0495. Treat this as a hypothesis for the next registered
test, not as a finding confirmed here.

## 4. Findings outside the run

These come from reading the code and documents, plus the classifier inspection in
`classifier_check.json`.

1. **The evidence classifier fires on substrings.** The keyword `rct` labels 50 documents as RCTs,
   and 46 of them contain neither the word "rct" nor a trial registration. They match
   "infarction" (33), "infarctions" (7), "antarctic" (3), "arctic" (2), "hrct" and others.
   Separately, 2,510 of 3,633 documents match no keyword at all. Those are assigned
   `UNCONTROLLED_OBSERVATIONAL` (weight 0.45) by default rather than treated as unknown. The
   August report called the classifier a proxy, which it is, but it is also mis-firing. Because
   the evidence weight drives the reranking (D5), this matters for every future run.
2. **The "Type A" parameters have no source.** `FRAMEWORK.md` says the evidence weights
   (0.85/0.65/0.45/0.25, with 95% CIs) come from "meta-epidemiological studies comparing effect
   estimates across study designs", and that the decay rates λ come from "citation half-life
   analysis". Neither cites a study or supplies the data. Meta-epidemiological studies estimate
   ratios of effect estimates, not reliability weights in [0, 1], and no mapping from one to the
   other is given. The best-known such review found "little evidence for significant effect
   estimate differences between observational studies and RCTs" (pooled ratio of odds ratios
   1.08; Anglemyer et al., 2014; since updated as Toews et al., 2024, which I could not read).
   A later review of 74 pooled pairs found no significant difference in 79.7% of them
   (Hong et al., 2021). On the evidence in this repository, these are Type B design choices and
   belong in the sensitivity analysis.
3. **`FRAMEWORK.md` shows an evaluation output that no code produces.** Its "Example Output" block
   reports `clinical: P@10=0.900 NDCG@10=0.820 … Conjecture test: PASS (+0.070 improvement)`.
   `evaluate.py` has no `__main__`, so `python -m clinical_ir.evaluate` prints nothing. The block
   contradicts the only real evaluation and has not been corrected. The README correction about
   "Implementation complete" did not reach `FRAMEWORK.md`, whose status section still ticks
   "Evaluation framework" and "Synthetic data testing" as completed.
4. **The registered outcomes disagree with one another.**
   - `default.json` names `decision_concordance` as the primary outcome, with refutation
     threshold 0.65 and a P@10 bar of 0.75.
   - `FRAMEWORK.md` lists four falsification criteria, including "sensitivity collapse". The
     README and the August report list three.
   - "≥5%" can be read as absolute (+0.05, as `evaluate.py` and `run_eval.py` implement it) or
     relative.
   - The registered data were "TREC CDS or custom annotation". The switch to NFCorpus was made
     before results were seen, but it is not logged as a deviation.

   None of this changes the criterion-1 verdict, which fails under every reading. Each needs
   settling before the next registered test.
5. **The August run cannot be reproduced from the repository as documented.**
   - `pip install -e .` installs only `evals`. The `clinical_ir` the August run imported is not
     recorded, and neither is the commit, package versions, encoder revision, data checksums or
     any per-query output.
   - The rerun matching to ten decimals shows the code was equivalent, but that is a finding
     about this rerun, not something the August record could show.
   - `run_eval.py` keys its embedding cache on model, tag and *count* only. A stale cache of
     the same size would load silently against different texts.
6. **The shipped tests cannot catch much.** Neither test file collects under `pytest` as the
   repository is laid out. The seven tests in `test_clinical_ir.py` pass once the package is
   importable from outside the repository directory, which is what the first commit's "All tests
   passing (7/7)" reflects. `test_relevance_bounded` asserts a score in [0, 1] on random vectors,
   which keep cosine near zero. For a systematic review with no action keywords and cosine −1,
   the score is 1 · 1 · (0.5 · (−1) + 0.3 · 0.5 + 0.2 · 0) = −0.35, so the function can leave
   [0, 1]. This test cannot fail on the inputs it uses.
7. **Prior art.** Reranking MEDLINE citations by evidence-based-medicine criteria (PICO match,
   strength of evidence), against a PubMed baseline, was published by Demner-Fushman and Lin
   (2007), who report significant gains. `FRAMEWORK.md`'s related work does not mention it. The
   epistemic-type classification of parameters may be new. The multidimensional clinical
   relevance function is not.

## 5. What holds up

- The falsification criteria were committed on 3 March 2026, the run is dated 23 August, and the
  registered primary criterion was applied as written.
- The headline result reproduces exactly, and the harness passes a positive control.
- Temporal and evidence criteria are reported as untested rather than passed, which is correct.
  Temporal-off is identical by construction, and the evidence labels are a proxy scored against
  judgments that do not reward evidence quality.
- The README's correction of "Implementation complete", and its refusal to re-run on a corpus
  chosen after the result, are sound.

## 6. Limitations of this audit

- One dataset, the same one. Every diagnostic describes this pipeline on NFCorpus and nothing
  wider.
- The data came from a mirror. Equivalence with the August files rests on exact reproduction,
  because the August run recorded no checksums.
- The ±0.03 equivalence margin is mine, set before the run. D1's interval excludes gains above
  0.0181 but not small ones, and the design's 80%-power detectable effect for D1 is 0.0136.
- E1 and E2 were run after the results were seen and are exploratory.
- The literature check was a handful of targeted searches, not a systematic review. I did not
  verify the August report's figure of ~0.325 for BEIR's published BM25 score on NFCorpus.

## 7. Recommended next steps

1. Leave the August record as it is, and add a dated correction to the README pointing here. The
   mechanism stated in `evals/report.md` §Observations and in Conjecture 3 of the companion note
   rests on an artefact.
2. Keep the registered next test (similarity-trained encoder for the topical component), and
   preregister these controls with it:
   - a random reordering of the same pool;
   - a random-vector encoder;
   - random tie-breaking for any configuration with few distinct scores;
   - BM25 reordering of the pool as a positive control;
   - an oracle-topic run of the composite (E2), to confirm the composite passes a strong signal.

   Save per-query outputs and a manifest.
3. Fix the classifier: match whole words, and give "no keyword" its own *unknown* level rather
   than defaulting to 0.45. Then rerun D5 to see how much ranking the evidence weight still
   controls.
4. Reclassify the evidence weights and λ values as Type B, or cite their derivations, and add them
   to the sensitivity analysis.
5. Remove or label the example output in `FRAMEWORK.md`. Reconcile `default.json`'s primary
   outcome with the falsification criteria, and decide between absolute and relative "5%".
6. Fix the package layout so `pip install -e .` provides `clinical_ir` and `pytest` collects.

## Provenance

| Reported number | Claim it supports | Source |
|---|---|---|
| 0.3103, 0.0705 and the other six August values | reproduction | `reproduction/results.json` `configs.*.ndcg@10`; `../results.json` |
| 0.551196 | scorer fidelity | `reproduction/run.log` |
| 0.2398 gap; −77.3% | size of failure | `../results.json` `delta_abs`, `delta_rel_pct` |
| +0.0083 [−0.0010, 0.0181], p = 0.091; 87/169/67 | D1 | `results.json` `D1_full_vs_random_rerank` |
| 0.0623 random; 0.3102 BM25 order of pool | D1, D6 | `results.json` `means.random_rerank`, `means.bm25_rerank_of_pool` |
| 73.7; 0.667; +0.00007; 305 ties; 0.3104 | D2 | `results.json` `D2_pool_composition`, `means.hybrid_unreranked` |
| 0.0119, 0.999, median 9 distinct scores; +0.000007 [−0.0044, 0.0044] | D3 | `results.json` `D3_topic_off_ties` |
| +0.0084 [−0.0034, 0.0206], p = 0.179 | D4a | `results.json` `D4a_cos_only_vs_random` |
| 0.0496, 0.0233, −0.0210 [−0.0295, −0.0134] | D4b | `results.json` `D4b_random_embeddings` |
| ρ 0.71 / 0.51 / 0.25 | D5 | `results.json` `D5_score_drivers` |
| −0.00006; 0.5488 | D6 | `results.json` `D6_controls` |
| +0.0119, p_Holm = 0.0004 | D7 | `results.json` `D7_ablation_paired_tests.abl_topic_off` |
| 0.2180; 0.0706 | D8 | `results.json` `D8_precision_at_10` |
| 405; 127; 25; 0.975 | E1 | `exploratory_results.json` `E1` |
| 0.4993; 0.5488; 0.0495 loss | E2 | `exploratory_results.json` `E2` (loss = ceiling − oracle_topic_in_R) |
| 50, 46, 33, 7, 3, 2; 2,510 of 3,633 | classifier | `classifier_check.json` |
| 0.0136 | D1 detectable effect | `results.json` `D1_full_vs_random_rerank.mde80` |
| 1.08; 79.7% | Type A weights | Anglemyer et al. (2014); Hong et al. (2021), as retrieved |

## References

Anglemyer, A., Horváth, H., & Bero, L. A. (2014). Healthcare outcomes assessed with observational
study designs compared with those assessed in randomized trials. *Cochrane Database of Systematic
Reviews*, 2014(4). https://doi.org/10.1002/14651858.MR000034.pub2 (superseded by the 2024 update
below).

Demner-Fushman, D., & Lin, J. (2007). Answering clinical questions with knowledge-based and
statistical techniques. *Computational Linguistics*, 33(1), 63–103.
https://doi.org/10.1162/coli.2007.33.1.63

Hong, Y. D., Jansen, J. P., Guerino, J., et al. (2021). Comparative effectiveness and safety of
pharmaceuticals assessed in observational studies compared with randomized controlled trials.
*BMC Medicine*, 19(1). https://doi.org/10.1186/s12916-021-02176-1

Toews, I., Anglemyer, A., Nyirenda, J. L. Z., et al. (2024). Healthcare outcomes assessed with
observational study designs compared with those assessed in randomized trials: A
meta-epidemiological study. *Cochrane Database of Systematic Reviews*, 2024(1).
https://doi.org/10.1002/14651858.MR000034.pub3 (retrieved as metadata only; findings not read).

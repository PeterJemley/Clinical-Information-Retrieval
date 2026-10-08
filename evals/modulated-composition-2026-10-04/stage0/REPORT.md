# Stage 0 pilot: results (exploratory)

**Date:** 2026-10-08
**Plan:** [`../PREREGISTRATION.md`](../PREREGISTRATION.md), with Amendment 1 committed in
`4dbc7ea` before any Stage 0 data were accessed.
**Script:** `stage0.py`, committed in `5bbf58a` before it ran. The run manifest's script
checksum matches that commit.
**Outputs:** `results.json`, `per_query.csv`, `manifest.json`, `run.log`.

**Stage 0 checks the plumbing and the controls. It is never evidence for the claim.** As
the plan requires, nothing below changes M, the gate table, the modules or the decision
rule.

## Verdict

**The harness works, and the controls behave as they should.**
- Every fidelity check passes. The scaled-up BM25 reproduces the repository's BM25 exactly.
- The artificial-signal control is detected.
- The noise control does not reach the bar.
- One of the three registered predictions missed: noise modules hurt more than I predicted.

**NFCorpus cannot exercise the modulation.** The registered rule puts 304 of the 324 dev
queries in the "other" column. Because evidence and temporal are inert here, every
difference between M and U comes from one module, actionability, switched off for most
queries. The significant gains of M over U and over random gate tables therefore show only
that the actionability keywords are noise on this corpus. They are not evidence that
conditioning on question type helps. This is the attribution problem that control V was
added to Stage 1 to catch, there for the evidence module.

## Fidelity checks

| Check | Result |
|---|---|
| A5 publication-type mapping, hand-written cases | 10 of 10 pass |
| A6 sparse BM25 against the repository's BM25, test queries | top 100 identical on 323 of 323; NDCG@10 0.3103001667, difference 0.0 |
| B0 equals the repository's BM25 top 100, dev queries | 324 of 324 |
| M with every gate but BM25's off equals B0 | 324 of 324 |
| V equals U (evidence is inert on NFCorpus) | 324 of 324 |
| F equals version 0.1's formula computed independently | 0 mismatches |

## Results on 324 dev queries

| Configuration | NDCG@10 |
|---|---|
| B0: BM25 order | 0.2683 |
| M: modulated composition | 0.2937 |
| U: same design, all modules on | 0.2631 |
| V: U with evidence off | 0.2631 |
| L: learned ranker | 0.2956 |
| F: version 0.1 on the BM25 pool (A7) | 0.0508 |
| N̄: noise modules, mean of 5 seeds | 0.2348 |
| R̄: shuffled gate tables, mean of 5 seeds | 0.2832 |
| P̄: M plus an artificial informative module, mean of 5 seeds | 0.3445 |

Paired differences use 95% bootstrap intervals and sign-flip p-values. The Holm correction
covers the five comparisons of M.

| Comparison | Mean | 95% CI | p (Holm) |
|---|---|---|---|
| **M − B0 (primary)** | +0.0254 | [+0.0164, +0.0351] | < 0.001 (uncorrected) |
| M − F | +0.2429 | [+0.2135, +0.2731] | 0.0005 |
| M − U | +0.0307 | [+0.0224, +0.0393] | 0.0005 |
| M − V | +0.0307 | [+0.0224, +0.0393] | 0.0005 |
| M − L | −0.0018 | [−0.0064, +0.0031] | 0.47 |
| M − R̄ | +0.0106 | [+0.0066, +0.0145] | 0.0005 |
| N̄ − B0 (control) | −0.0335 | [−0.0417, −0.0255] | < 0.001 (uncorrected) |
| P̄ − M (control) | +0.0507 | [+0.0433, +0.0585] | < 0.001 (uncorrected) |

Applying the registered decision rule gives **Refuted**, because the interval for M − B0
lies entirely below +0.05. That verdict is reported only because the plan asks for it. On
a corpus where the modulation barely varies and the evidence and temporal modules are
inert, it says nothing about the claim.

## Registered predictions

| Prediction | Outcome |
|---|---|
| \|M − B0\| < 0.03 | **Held** (+0.0254) |
| \|N̄ − B0\| < 0.03 | **Failed** (−0.0335) |
| P̄ − M > 0, interval lower bound > 0 | **Held** (+0.0507, [+0.0433, +0.0585]) |

**The failed prediction.** Adding noise modules at equal weight to BM25 cost 0.0335, not
less than 0.03 as I predicted. That is a misjudged magnitude, not a broken control. The
control does what a must-fail control should: it falls below BM25, and its interval stays
well under the +0.05 bar the decision rule checks. No change follows from it.

## What the pilot shows about the design

1. **Question types barely vary on NFCorpus.** The registered rule gives 304 "other", 16
   treatment, 3 prognosis and 1 diagnosis/test. NFCorpus queries are mostly topic titles,
   not clinical questions. NFCorpus would also fail Amendment 1's sixth criterion as a
   Stage 1 corpus.
2. **The gains from gating come from one module.**
   - For "other" and prognosis queries, M switches actionability off and U keeps it on.
     With evidence and temporal inert, that is the only difference between them.
   - The shuffled tables tell the same story. The seeds that send "other" queries to a
     column with actionability off score 0.2957 (seeds 0 and 4), or equal M (seed 1, which
     drew the identity permutation). Those that send them to a column with actionability on
     score 0.2653 (seeds 2 and 3).
   - So on this corpus, M − U and M − R̄ measure removing the actionability keywords, not
     conditioning on question type.
3. **For Stage 1, V is the matching control.** On the Clinical Decision Support topics,
   the diagnosis/test and treatment columns differ only in the evidence module, so V
   isolates that module. The general lesson is to compare M with U minus every module the
   gate table switches off. Stage 1's corpus needs that only for evidence.
4. **The designed composition matches the learned ranker.** M has no fitted parameters and
   is indistinguishable from L, which was tuned on 2,022 training queries. L's
   cross-validation picked the smallest penalty in the registered grid (0.01), a boundary
   value, so a wider grid might help L. The grid is left as registered. Widening it would
   only strengthen the baseline, so it is an option to log as a deviation before Stage 1.
5. **M beats BM25 by about 0.025 here.** On "other" queries, M is BM25 plus MedCPT with
   equal weight, so this is the fusion gain. It is consistent with the topical test (MedCPT
   alone +0.020 over BM25 on the test queries) and below the 0.05 bar.

## Provenance

| Number | Source |
|---|---|
| Configuration means | `results.json` → `means` |
| Paired comparisons | `results.json` → `primary_M_minus_B0`, `secondary_holm`, `controls` |
| Fidelity checks | `results.json` → `fidelity` |
| Question-type counts | `results.json` → `question_types_dev` |
| L penalty and training queries | `results.json` → `L` |
| R permutations | `results.json` → `R_permutations` |
| Verdict and predictions | `results.json` → `decision_rule_applied_exploratory`, `stage0_predictions` |
| Data, encoder and embedding checksums; timings | `manifest.json` |

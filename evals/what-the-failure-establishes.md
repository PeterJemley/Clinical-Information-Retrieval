# What the evaluation established, and what it did not

**Companion to `evals/report.md`, 2026-08-23.**

This note does not revise the result. The registered criterion — at least 5% NDCG@10
over BM25 — **failed**, at 0.071 against 0.310, with non-overlapping intervals over 323
queries. That stands, and nothing below is offered to soften it.

What follows is a different question: *what does that reading actually tell us?*

---

## The apparatus, not the verdict

Switching on a newly built instrument and getting a wrong reading is the normal
condition of experimental work, not a verdict on the theory it was built to test. The
reading poses a problem. The problem is then attacked the same way any problem is — by
conjecture and criticism — and the conjectures worth forming include the instrument, the
calibration, the test bench, and the measured quantity, not only the theory.

Being clever about what could have gone wrong is the whole skill. So: five conjectures
that would each produce this reading, and what the evidence says about each.

---

## Conjecture 1 — The theory is wrong

*Multidimensional relevance weighting does not help. Topical relevance is sufficient in
this domain.*

This is the conjecture the pre-registration was built to test. It is also, on the
present evidence, **the least supported of the five** — not because the result was kind
to it, but because conjectures 2 through 5 each independently account for the reading,
and three of the four dimensions never entered the measurement at all.

**Status: barely tested.**

---

## Conjecture 2 — The instrument was incomplete

*The apparatus was missing a component when it was switched on.*

`default.json` specifies PubMedBERT as the encoder. No code in the repository loaded any
embedding model; the module self-tests ran on `np.random.randn` vectors. The evaluation
harness supplied the missing embedding step in order to run at all.

A theory cannot be refuted by an apparatus that was not finished. The repository's status
line said implementation was complete. It was not, and that is the most consequential
thing this evaluation surfaced — it was invisible until something tried to use the thing
end to end.

**Status: confirmed. Correct the status line.**

---

## Conjecture 3 — The instrument is miscalibrated

*A component is present but measuring badly, and the composite inherits the fault.*

Mean-pooled BERT embeddings are poorly calibrated for cosine similarity without
similarity fine-tuning. The topic-off ablation (0.0824) **outperforms** topic-on
(0.0705): removing the topical signal improves the system. The candidate pool is itself
60% weighted toward that signal, which is then reranked by a score dominated by the same
signal — so the fault compounds.

This is an instrument fault, not a theory fault. The theory says combine topical with
temporal and evidence. If the topical measurement is broken, every combination
containing it is broken.

**Status: strongly supported, and it is the dominant driver.**

---

## Conjecture 4 — The test bench cannot measure the quantity

*The dimensions the theory is about were not present in the data.*

- **Temporal:** NFCorpus carries no publication dates. τ was neutralised to 1.0 across
  all 3,633 documents. The temporal-off ablation is bit-identical to the full system
  because there was nothing to ablate.
- **Evidence:** no study-type metadata. Levels were assigned by keyword heuristic — a
  proxy, disclosed as such, measuring the heuristic rather than evidence hierarchy.
- **Applicability:** no population metadata. Constant 0.5 for every document, therefore
  no ranking effect at all.

Three of four distinctive dimensions were inert or unmeasurable. The experiment tested
the topical reranking path and very little else.

**Status: confirmed.**

---

## Conjecture 5 — The dependent variable is wrong

*The ground truth does not reward what the theory optimises.*

NFCorpus relevance judgments reward topical relevance to nutrition and medical questions.
Nothing in the gold labels rewards ranking a randomised trial above a case series. A
perfect evidence-hierarchy signal would have had nothing to improve against, and would
have registered as noise or worse.

This is not a defect of NFCorpus. It is a mismatch between the benchmark's question and
the framework's.

**Status: confirmed.**

---

## What follows

Two statements, both true, and not in tension:

1. **The registered criterion failed.** The system as specified, on a standard benchmark,
   performs far below a lexical baseline. That is recorded and stands.
2. **The multidimensional thesis remains largely untested.** It was not given an
   apparatus capable of testing it.

The discipline that matters now is not to let the second sentence quietly retract the
first. It does not. The system was evaluated as specified and it lost.

**The next test is not a re-run.** Fixing the instrument and running again on a corpus
chosen after seeing this result would be dataset-shopping — the same failure as
post-hoc parameter tuning, wearing different clothes.

The next step is to isolate conjecture 3, because it is cheap and it discriminates:
measure the topical component alone, with a similarity-trained encoder in place of mean
pooling, against BM25 on the same data. If it remains worse, the semantic path is the
problem. If it beats BM25, the instrument fault is isolated and the composite becomes
worth testing.

Only then does the full thesis become testable, and only on a corpus carrying publication
dates, gold study-type labels, and judgments sensitive to recency and evidence quality.
That test should be registered before it is run, exactly as this one was.

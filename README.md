# Clinical Information Retrieval Framework

A falsifiable framework for domain-specific information retrieval that treats relevance as multidimensional.

## The Conjecture

Standard IR systems optimize for topical relevance. In specialized domains, this is necessary but insufficient. A document may be topically relevant yet temporally obsolete, evidentially weak, population-mismatched, or non-actionable.

This framework conjectures that multidimensional relevance scoring will outperform topical matching.

## Epistemic Stance

We distinguish three types of claims:

| Type | Description | Constraint |
|------|-------------|------------|
| **A** | Theoretically derived | Fixed by external evidence |
| **B** | Informed design choices | Subject to sensitivity analysis |
| **C** | Bold conjectures | Test and report |

## Modules

| Module | Purpose |
|--------|---------|
| `temporal.py` | Domain-specific temporal decay (Type A) |
| `evidence.py` | Evidence hierarchy weights (Type A) |
| `relevance.py` | Core relevance function R(d,q) |
| `retrieval.py` | BM25 baseline + hybrid search |
| `evaluate.py` | Metrics and conjecture testing |
| `default.json` | All parameters with epistemic types |

## Relevance Function

```
R(d, q) = τ(d,t) · w_e(d) · Σ αᵢ · rᵢ(d,q)
```

Where:
- τ = temporal decay (Type A - from citation half-life analysis)
- w_e = evidence weight (Type A - from meta-epidemiological studies)
- α = component weights (Type B - sensitivity analysis)
- r = component scores (topical, applicability, actionability)

## Falsification Criteria

The framework is refuted if:

1. Clinical relevance does not beat BM25 by ≥5% on NDCG@10
2. Temporal decay does not improve vs. recency-agnostic baseline
3. Evidence weighting does not improve vs. uniform weights

"Equal to" counts as failure—if complex scoring merely matches simple baseline, the complexity is not justified.

## Usage

```python
from clinical_ir.relevance import Document, Query, clinical_relevance
from datetime import datetime
import numpy as np

doc = Document(
    id="pmid_12345",
    title="Randomized trial of restrictive transfusion thresholds",
    abstract="RCT comparing 70 g/L vs 100 g/L...",
    content_embedding=np.random.randn(768),
    publication_date=datetime(2020, 1, 1),
    population={"age_min": 50, "age_max": 80},
)

query = Query(
    text="transfusion threshold cardiac surgery",
    embedding=np.random.randn(768),
    population={"age_min": 65, "age_max": 85},
)

score = clinical_relevance(doc, query)
```

## Status

**Evaluated 2026-08-23. The primary falsification criterion failed.**

On BEIR/NFCorpus (3,633 documents; 323 test queries with graded human relevance
judgments), the full clinical relevance function scored **NDCG@10 = 0.071** against a
**BM25 baseline of 0.310** — 77% below the baseline, with non-overlapping 95%
confidence intervals. The registered bar was ≥5% *above* BM25. By this framework's own
criterion, that "equal to counts as failure," the core conjecture is refuted on this
dataset. It did not narrowly miss; it lost heavily to the simple baseline it was built
to beat.

The other two criteria **could not be evaluated on this dataset**, and remain untested
rather than passed:

- **Temporal decay.** NFCorpus carries no publication dates, verified across all 3,633
  documents. τ was neutralised to 1.0 throughout, so the temporal ablation is identical
  to the full system. There was no temporal signal to test.
- **Evidence weighting.** NFCorpus carries no study-type metadata. Evidence levels were
  assigned by this repository's own keyword classifier — a proxy, which measures the
  heuristic rather than the evidence hierarchy. NFCorpus relevance judgments also reward
  topical relevance, so a perfect evidence signal would have had no ground truth to
  improve against.

### Correction to a prior claim in this file

This section previously read "Implementation complete." That was inaccurate.
`default.json` names PubMedBERT as the encoder, but no code in this repository loaded any
embedding model; the module self-tests ran on `np.random.randn` vectors. The evaluation
harness had to supply the missing embedding step in order to run at all. The
implementation was incomplete at the time the falsification criteria were registered,
and that was not visible until something attempted to use it end to end.

Two further defects surfaced by the run, not modified: the shipped test suite does not
collect under `pytest` (relative imports against a repo-root `__init__.py`), and
`tests/test_evaluate.py` does a bare `import evaluate`.

**A second correction, 26 August 2026.** This README, `FRAMEWORK.md` and the companion
note each described a parameter file named `config.json`. The repository has never
contained one. The file is `default.json`, and no code here opens it under either name.
This is the defect recorded above, one level further down: the configuration was written
as a design and never connected to anything, its filename included. The documents that
describe the system are corrected. `evals/report.md` and `evals/run_eval.py` record what
was run on 23 August and are left as written, because the record is not edited after the
run.

### Where the results are

- Full results, ablations, and the components that could not be evaluated:
  [`evals/report.md`](evals/report.md)
- A companion note separating what the result establishes from what it does not:
  [`evals/what-the-failure-establishes.md`](evals/what-the-failure-establishes.md)

### On authorship

Three of the commits in this repository name an AI coding agent as co-author. The
division of work is visible in the log.

The framework and the criteria that would refute it are two commits from March 2026 —
`b27b17e` on the 3rd and `b01a3be` on the 26th, 1,260 lines added and none removed,
neither carrying a co-author. Those two commits are the whole of the system under test:
the relevance function, the temporal and evidence modules, the retrieval baseline, the
parameter file, and `FRAMEWORK.md`, which states the falsification criteria.

The agent's three commits are dated 23 and 26 August. They add `evals/` and edit this
README. They modify no file that defines the framework — not `relevance.py`,
`temporal.py`, `evidence.py`, `retrieval.py`, or the parameter file. The system under
test and the apparatus that tested it were written nearly six months apart, and the
second did not alter the first.

Also mine: the instruction to run the evaluation and publish the outcome whichever way it
came out, and the decision against re-running on a corpus chosen after seeing the result.

Execution earned a finding of its own. The run exposed a defect that inspection had not —
a language model named in the configuration that no code in this repository ever loaded.
Defects of that kind stay invisible until something attempts to use the thing end to end.

### What comes next

Not a re-run on a more favourable dataset. Choosing a corpus after seeing this result
would be selection by outcome, and would void the value of having registered the
criterion in advance.

The next test isolates the most strongly supported explanation: the topical component
alone, with a similarity-trained encoder in place of mean pooling, measured against BM25
on the same data. If it still loses, the semantic path is the problem. If it wins, the
instrument fault is isolated and the composite becomes worth testing. That test will be
registered before it is run.

## License

MIT

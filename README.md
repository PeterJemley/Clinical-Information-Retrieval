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
| `config.json` | All parameters with epistemic types |

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

Implementation complete. Awaiting evaluation on real clinical relevance judgments (TREC Clinical Decision Support or custom annotation).

## License

MIT

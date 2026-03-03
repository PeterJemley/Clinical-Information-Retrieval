# Multidimensional Relevance for Domain-Specific Information Retrieval

## A Falsifiable Framework

**Status**: Draft (work in progress)  
**Domain**: Information Retrieval  
**Author**: Peter Jemley  

---

## Plain Language Summary

Standard search engines answer one question: *Does this document match my query terms?*

In specialized domains—medicine, law, engineering—users implicitly ask more:

1. **Is this current?** A 2005 guideline may be superseded.
2. **Is this trustworthy?** A case report provides weaker evidence than a randomized trial.
3. **Does this apply to my situation?** A study of young adults may not generalize.
4. **Does this help me decide?** Background knowledge differs from actionable recommendations.

This project proposes a framework that scores documents on multiple relevance dimensions. We make specific, falsifiable claims about what dimensions matter, commit to testing those claims, and specify in advance what observations would prove us wrong.

---

## The Core Conjecture

**Topical relevance is necessary but insufficient for domain-specific retrieval.**

A retrieval system optimized only for topical matching will systematically fail users who need to evaluate evidence quality, temporal validity, population applicability, and decision utility.

We conjecture that a multidimensional relevance function will outperform topical baselines on domain-specific retrieval tasks.

---

## Epistemic Framework

### The Problem with Most IR Research

Most retrieval frameworks present parameters as if derived from first principles, then adjust them until evaluation metrics look good. This is epistemologically backwards—the "derivations" serve as post-hoc rationalizations for values chosen to fit the data.

We take a different approach: explicitly classifying every claim by its epistemic status.

### Three Types of Claims

| Type | Description | What Constrains It | What Happens If Wrong |
|------|-------------|-------------------|----------------------|
| **A** | Theoretically derived | External evidence | Contradicts source data |
| **B** | Informed design choice | Sensitivity analysis | Framework fragile to changes |
| **C** | Bold conjecture | Empirical test | Framework refuted |

### Type A: Theoretically Derived Parameters

These are constrained by external evidence. Changing them requires that evidence to be wrong.

**Temporal decay rates**: Derived from bibliometric citation half-life analysis. Different domains have different rates of knowledge turnover:

| Domain | λ (decay rate) | Half-life | Mechanism |
|--------|----------------|-----------|-----------|
| Infectious disease | 0.20 | 3.5 years | Pathogens mutate, resistance emerges |
| Oncology | 0.15 | 4.6 years | Treatment advances rapidly |
| Cardiology | 0.10 | 6.9 years | Physiology stable, guidelines evolve |
| Surgery | 0.08 | 8.7 years | Techniques evolve slowly |
| Anatomy | 0.02 | 34.7 years | Human body doesn't change |

These rates are hard to vary because the mechanisms constrain them. Giving anatomy the same decay rate as infectious disease would produce absurd results—discounting valid anatomical knowledge that hasn't changed.

**Evidence hierarchy weights**: Derived from meta-epidemiological studies that compared effect estimates across study designs. When RCTs and observational studies examine the same question, the weights reflect empirically observed reliability differences.

| Study Design | Weight | 95% CI |
|--------------|--------|--------|
| Systematic review | 1.00 | — |
| RCT | 0.85 | 0.78–0.92 |
| Controlled observational | 0.65 | 0.55–0.75 |
| Uncontrolled observational | 0.45 | 0.35–0.55 |
| Expert opinion | 0.25 | 0.15–0.35 |

### Type B: Informed Design Choices

These have rationale but not unique determination. Different reasonable people might choose differently.

| Parameter | Default | Sensitivity Range | Rationale |
|-----------|---------|-------------------|-----------|
| Hybrid search β | 0.4 | 0.2–0.8 | Balance keyword precision with semantic recall |
| Component weights | (0.5, 0.3, 0.2) | Full grid | Topic primary, then applicability, then actionability |
| Redundancy λ | 0.3 | 0.1–0.5 | Moderate diversity without sacrificing relevance |

**Sensitivity analysis reveals which choices matter.** If performance is robust across a wide range, the specific value isn't doing explanatory work. If performance collapses outside a narrow range, that reveals framework fragility.

### Type C: Bold Conjectures

These cannot be derived theoretically. The necessary evidence doesn't exist until we run the experiments.

- [ ] Multidimensional relevance outperforms BM25 baseline
- [ ] Temporal decay weighting improves retrieval vs. recency-agnostic
- [ ] Evidence hierarchy weighting improves retrieval vs. uniform weights
- [ ] Framework transfers to domains beyond initial test case

**We commit to reporting results regardless of outcome.**

---

## The Relevance Function

```
R(d, q) = τ(d,t) · w_e(d) · Σ αᵢ · rᵢ(d,q)
```

Where:

- **τ(d,t)** = temporal decay: `exp(-λ · (t - t_d))`
- **w_e(d)** = evidence weight based on study design
- **α** = component weights (Type B)
- **r** = component relevance scores

### Component Scores

**Topical relevance** `r_topic`: Cosine similarity of document and query embeddings (domain-specific encoder like PubMedBERT).

**Applicability** `r_apply`: Population overlap between document and query. A study of "adults 50-70" has 0.67 applicability to a query about "adults 45-75" (20-year overlap / 30-year query range). This continuous scoring avoids discarding partially applicable evidence.

**Actionability** `r_action`: Learned classifier on features indicating decision support value (dosing recommendations, numerical thresholds, explicit recommendations, etc.).

### Why These Dimensions?

Each dimension exists because removing it breaks something specific:

| Dimension | What breaks without it |
|-----------|----------------------|
| Temporal | Cannot distinguish current from obsolete evidence |
| Evidential | Cannot distinguish strong from weak grounds |
| Applicability | Retrieves rigorous studies of wrong populations |
| Actionability | Retrieves background info instead of decision support |

**Why not other dimensions?** Possible additions (reading level, source prestige, geographical relevance) don't correspond to distinct information needs that the four dimensions don't already capture.

---

## Architecture

### Three-Tier Structure

Each tier solves a problem the others cannot:

**Tier 1: Evidence Aggregation**
- Retrieves candidates from corpus of millions
- Must be fast, must balance precision/recall
- Implementation: Hybrid search (BM25 + semantic)

**Tier 2: Guideline Synthesis** (domain-specific)
- Identifies and reconciles authoritative documents
- Detects conflicts, prioritizes by recency/authority
- Cannot merge with Tier 1: different optimization objectives

**Tier 3: Context Adaptation**
- Adapts general recommendations to specific situations
- Applies threshold adjustments, flags exceptions
- Cannot merge with Tier 2: requires situation-specific data

**Why three tiers?** Merging any two leaves a problem unsolved. Merging 1+2 makes retrieval too slow. Merging 2+3 makes synthesis context-dependent. Merging 1+3 skips conflict reconciliation.

---

## Falsification Criteria

The framework is refuted if:

1. **Multidimensional ≤ BM25**: If NDCG@10 improvement < 5%, complexity not justified
2. **Temporal decay doesn't help**: If recency-agnostic baseline matches or exceeds
3. **Evidence weighting doesn't help**: If uniform weights match or exceed
4. **Sensitivity collapse**: If performance degrades sharply outside narrow parameter ranges

Note: "equal to" counts as failure. If complex scoring merely matches simple baseline, the complexity isn't justified.

---

## The Moratorium Problem

A crucial insight from the history of medical research:

> "When early efforts at heart transplantation ran into trouble, Britain embraced a self-imposed moratorium—sensible given the likely outcomes of early operations, but cowardly and disingenuous since we avoided doing experimental human surgery while waiting for others to do so instead."

Britain waited for others to take risks and generate knowledge, then benefited without contributing.

**The epistemological point**: Demanding complete theoretical derivation before attempting anything new can be intellectual cowardice disguised as rigor. Some knowledge can only be created by trying.

This is why we distinguish Type A parameters (genuinely derivable) from Type C conjectures (must be tested). We don't pretend all parameters can be derived a priori. We don't refuse to test until derivation is complete.

---

## Implementation

### Module Structure

```
clinical_ir/
├── config.json      # All parameters with epistemic types
├── temporal.py      # Temporal decay (Type A)
├── evidence.py      # Evidence hierarchy (Type A)
├── relevance.py     # Core R(d,q) function
├── retrieval.py     # BM25 baseline + hybrid search
├── evaluate.py      # Metrics + conjecture testing
└── README.md
```

### Running

```bash
# Test individual modules
python -m clinical_ir.temporal
python -m clinical_ir.evidence
python -m clinical_ir.relevance
python -m clinical_ir.retrieval
python -m clinical_ir.evaluate
```

### Example Output

```
Temporal Decay (infectious disease, λ=0.20):
   0 years ago: τ = 1.000
   5 years ago: τ = 0.368
  10 years ago: τ = 0.136

Evidence Weights:
  SYSTEMATIC_REVIEW: 1.00
  RCT: 0.85 (95% CI: 0.78-0.92)
  CONTROLLED_OBSERVATIONAL: 0.65 (95% CI: 0.55-0.75)

Evaluation:
  bm25    : P@10=0.800 NDCG@10=0.750 MRR=1.000
  clinical: P@10=0.900 NDCG@10=0.820 MRR=1.000
  Conjecture test: PASS (+0.070 improvement)
```

---

## What We Will Learn

### If the Framework Succeeds

- Which dimensions contribute most (sensitivity analysis)
- Parameter robustness ranges
- Whether principles transfer across domains
- Practical guidance for domain-specific retrieval

### If the Framework Fails

- Which specific conjectures failed
- What assumptions were violated
- What alternative approaches might work
- Published negative results (scientifically valuable)

### Knowledge Requiring Empirical Test

These questions cannot be answered by theoretical analysis:

- Does multidimensional scoring beat topical baselines?
- How sensitive is performance to component weights?
- What are the minimum data requirements for transfer?

Refusing to attempt the research guarantees these questions remain unanswered.

---

## Related Work

This framework builds on:

- **Learning to rank**: Uses relevance dimensions as features, but fixes some parameters theoretically rather than learning all from data
- **Domain-specific IR**: TREC Clinical Decision Support, medical QA systems
- **Evidence-based medicine**: Systematic approach to evidence hierarchy
- **Citation analysis**: Bibliometric approaches to temporal validity

Key distinction: We explicitly classify parameters by epistemic type rather than treating all as learnable hyperparameters.

---

## Status and Next Steps

### Completed

- [x] Core relevance function implementation
- [x] BM25 baseline
- [x] Hybrid search
- [x] Evaluation framework
- [x] Synthetic data testing

### In Progress

- [ ] Real evaluation data (TREC CDS or custom annotation)
- [ ] Sensitivity analysis on Type B parameters
- [ ] Transfer experiments to second domain

### Future

- [ ] Tier 2 (guideline synthesis) implementation
- [ ] Tier 3 (context adaptation) implementation
- [ ] User study on decision quality

---

## Appendix: Mathematical Formalization

### Document Representation

Document d ∈ D with fuzzy membership in three spaces:
- μ_G(d) ∈ [0,1]: membership in guideline space
- μ_E(d) ∈ [0,1]: membership in evidence space
- μ_P(d) ∈ [0,1]: membership in protocol space

### Temporal Decay

τ(d, t) = exp(-λ_domain · (t - t_d))

where λ_domain derived from: λ = -(1/T) · ln(citations_T / citations_0)

### Evidence Weight

w_e(d) = Σ_l P(level=l | d) · weight(l)

where weights from meta-epidemiological studies with 95% CIs.

### Relevance Function

R(d, q) = τ(d, t) · w_e(d) · [α_topic · r_topic(d,q) + α_apply · r_apply(d,q) + α_action · r_action(d,q)]

subject to: Σ α_i = 1, α_i > 0

### Retrieval Optimization

S* = argmax_{S⊆D, |S|=k} Σ_{d∈S} R(d,q) - λ_red · Σ_{d,d'∈S} sim(d,d')

Submodular optimization with redundancy penalty.

---

*Draft document. Work in progress. Comments welcome.*

"""
Temporal relevance decay for clinical information retrieval.

Type A Parameter: Decay rates derived from citation half-life analysis.
These are constrained by external bibliometric evidence and should not
be adjusted post-hoc.
"""

import math
from datetime import datetime
from typing import Optional

# Domain-specific decay rates (Type A - theoretically derived)
# Derivation: λ = -(1/T) * ln(citations_at_T / citations_at_0)
DECAY_RATES = {
    "infectious_disease": 0.20,  # Half-life: 3.5 years
    "oncology": 0.15,            # Half-life: 4.6 years
    "cardiology": 0.10,          # Half-life: 6.9 years
    "pbm": 0.12,                  # Half-life: 5.8 years (default)
    "surgery": 0.08,             # Half-life: 8.7 years
    "anatomy": 0.02,             # Half-life: 34.7 years
}

DEFAULT_DECAY = 0.12  # PBM default


def temporal_decay(
    document_date: datetime,
    query_date: Optional[datetime] = None,
    domain: str = "pbm",
    lambda_override: Optional[float] = None
) -> float:
    """
    Calculate temporal relevance decay.
    
    τ(d, t) = exp(-λ * (t - t_d))
    
    Why this formula: Medical knowledge has a half-life that varies by domain.
    Infectious disease evidence ages fast (pathogens mutate, resistance emerges).
    Anatomical knowledge stays valid (human anatomy doesn't change).
    
    The decay rates are hard to vary because they're derived from citation
    analysis - changing them would contradict the bibliometric evidence.
    
    Args:
        document_date: Publication date of document
        query_date: Time of query (defaults to now)
        domain: Clinical domain for domain-specific decay rate
        lambda_override: Override decay rate (for sensitivity analysis only)
    
    Returns:
        Temporal relevance score in [0, 1]
    """
    if query_date is None:
        query_date = datetime.now()
    
    # Time difference in years
    delta_years = (query_date - document_date).days / 365.25
    
    # Can't have negative time (document from future)
    if delta_years < 0:
        delta_years = 0
    
    # Get decay rate
    if lambda_override is not None:
        decay_rate = lambda_override
    else:
        decay_rate = DECAY_RATES.get(domain.lower(), DEFAULT_DECAY)
    
    # Exponential decay
    return math.exp(-decay_rate * delta_years)


def half_life(domain: str) -> float:
    """
    Return the half-life in years for a domain.
    
    Half-life = ln(2) / λ
    """
    decay_rate = DECAY_RATES.get(domain.lower(), DEFAULT_DECAY)
    return math.log(2) / decay_rate


def years_until_threshold(
    threshold: float,
    domain: str = "pbm"
) -> float:
    """
    Calculate years until temporal relevance falls below threshold.
    
    Solving: threshold = exp(-λ * t)
    t = -ln(threshold) / λ
    """
    decay_rate = DECAY_RATES.get(domain.lower(), DEFAULT_DECAY)
    return -math.log(threshold) / decay_rate


# Self-test
if __name__ == "__main__":
    from datetime import timedelta
    
    now = datetime.now()
    
    print("Temporal Decay Module - Self Test")
    print("=" * 50)
    
    # Test decay over time for different domains
    for domain in ["infectious_disease", "anatomy", "pbm"]:
        print(f"\n{domain.upper()} (λ={DECAY_RATES[domain]}, half-life={half_life(domain):.1f}y)")
        for years_ago in [0, 1, 5, 10, 20]:
            doc_date = now - timedelta(days=years_ago * 365)
            decay = temporal_decay(doc_date, now, domain)
            print(f"  {years_ago:2d} years ago: τ = {decay:.3f}")
    
    print("\n" + "=" * 50)
    print("Years until τ < 0.5 (half-life check):")
    for domain in DECAY_RATES:
        hl = half_life(domain)
        check = years_until_threshold(0.5, domain)
        print(f"  {domain}: {hl:.1f} years (check: {check:.1f})")

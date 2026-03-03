"""
Evidence hierarchy weights for clinical information retrieval.

Type A Parameter: Weights derived from meta-epidemiological studies
comparing effect estimates across study designs.
"""

from typing import Optional
from enum import IntEnum


class EvidenceLevel(IntEnum):
    SYSTEMATIC_REVIEW = 1
    RCT = 2
    CONTROLLED_OBSERVATIONAL = 3
    UNCONTROLLED_OBSERVATIONAL = 4
    EXPERT_OPINION = 5


# Type A parameters - derived from meta-epidemiological studies
EVIDENCE_WEIGHTS = {
    EvidenceLevel.SYSTEMATIC_REVIEW: 1.00,
    EvidenceLevel.RCT: 0.85,
    EvidenceLevel.CONTROLLED_OBSERVATIONAL: 0.65,
    EvidenceLevel.UNCONTROLLED_OBSERVATIONAL: 0.45,
    EvidenceLevel.EXPERT_OPINION: 0.25,
}

WEIGHT_CONFIDENCE_INTERVALS = {
    EvidenceLevel.SYSTEMATIC_REVIEW: (1.00, 1.00),
    EvidenceLevel.RCT: (0.78, 0.92),
    EvidenceLevel.CONTROLLED_OBSERVATIONAL: (0.55, 0.75),
    EvidenceLevel.UNCONTROLLED_OBSERVATIONAL: (0.35, 0.55),
    EvidenceLevel.EXPERT_OPINION: (0.15, 0.35),
}

LEVEL_KEYWORDS = {
    EvidenceLevel.SYSTEMATIC_REVIEW: [
        "systematic review", "meta-analysis", "cochrane"
    ],
    EvidenceLevel.RCT: [
        "randomized", "randomised", "rct", "clinical trial"
    ],
    EvidenceLevel.CONTROLLED_OBSERVATIONAL: [
        "cohort", "case-control", "propensity"
    ],
    EvidenceLevel.UNCONTROLLED_OBSERVATIONAL: [
        "case series", "case report", "retrospective"
    ],
    EvidenceLevel.EXPERT_OPINION: [
        "expert opinion", "consensus", "editorial"
    ],
}


def evidence_weight(level: EvidenceLevel) -> float:
    return EVIDENCE_WEIGHTS.get(level, 0.5)


def classify_evidence_level(title: str, abstract: str) -> EvidenceLevel:
    text = f"{title} {abstract}".lower()
    for level in EvidenceLevel:
        for keyword in LEVEL_KEYWORDS.get(level, []):
            if keyword in text:
                return level
    return EvidenceLevel.UNCONTROLLED_OBSERVATIONAL


if __name__ == "__main__":
    print("Evidence Hierarchy Weights:")
    for level in EvidenceLevel:
        w = evidence_weight(level)
        ci = WEIGHT_CONFIDENCE_INTERVALS[level]
        print(f"  {level.name}: {w:.2f} (95% CI: {ci[0]:.2f}-{ci[1]:.2f})")

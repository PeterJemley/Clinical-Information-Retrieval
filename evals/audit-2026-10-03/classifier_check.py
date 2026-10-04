"""
Inspect what the evidence keyword classifier (evidence.classify_evidence_level) assigns on
NFCorpus, and how often a keyword fires only as a substring of another word.

    PYTHONPATH=<dir with clinical_ir symlink> python evals/audit-2026-10-03/classifier_check.py --data <nfcorpus dir>
"""
import argparse, collections, json, os, re
from clinical_ir.evidence import EvidenceLevel, LEVEL_KEYWORDS, classify_evidence_level

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    docs = [json.loads(l) for l in open(os.path.join(args.data, "corpus.jsonl"))]
    levels = collections.Counter()
    fired = collections.Counter()
    substring_only = collections.Counter()
    rct_words = collections.Counter()
    rct_docs_spurious = 0
    for r in docs:
        level = classify_evidence_level(r["title"], r["text"])
        levels[level.name] += 1
        text = f"{r['title']} {r['text']}".lower()
        hit = None
        for lvl in EvidenceLevel:
            hit = next((k for k in LEVEL_KEYWORDS[lvl] if k in text), None)
            if hit:
                break
        if hit is None:
            fired["(no keyword: default)"] += 1
            continue
        fired[hit] += 1
        if not re.search(r"\b" + re.escape(hit) + r"\b", text):
            substring_only[hit] += 1
        if hit == "rct":
            words = set(re.findall(r"\w*rct\w*", text))
            rct_words.update(words)
            if not ({"rct", "rcts"} & words or any(w.startswith("isrctn") for w in words)):
                rct_docs_spurious += 1
    out = {
        "n_docs": len(docs),
        "assigned_levels": dict(levels),
        "default_uncontrolled_observational_no_keyword": fired["(no keyword: default)"],
        "first_firing_keyword": dict(fired.most_common()),
        "substring_only_matches": dict(substring_only),
        "rct_keyword": {"docs": fired["rct"], "docs_without_rct_word_or_trial_registration": rct_docs_spurious,
                        "docs_containing_word": dict(rct_words.most_common())},
    }
    with open(os.path.join(HERE, "classifier_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

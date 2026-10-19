"""
Checks for stage1.py. They must pass before any Stage 1 data are accessed (Amendment 2, B4).

    python check_stage1.py        # exits non-zero if any check fails

1. Every branch of the decision rule, on constructed inputs, and the order of the rows.
2. Synthetic result sets whose verdicts are known in advance: one where every prediction
   holds (Supported), one where every prediction fails (Refuted), and two that must stop
   earlier (Test failed, Untestable).
3. The parsers, on hand-made files: NXML, topics in the 2014 and 2016 layouts, qrels and
   NCBI efetch XML.
4. The A5 publication-type mapping.
5. The sparse BM25 against the repository's BM25 on a synthetic corpus with ties.
6. The B2 manipulation check, and the scoring helpers.
7. A small end-to-end run on fixture data with the stand-in encoder: outputs, the B3 order,
   determinism, and the refusals (stand-in encoder outside test mode, a changed data file, a
   missing reference date, the A6 gate's target in a real run, judgments that cannot be read).

Everything is generated here; nothing is downloaded.
"""
import csv, json, os, shutil, subprocess, sys, tarfile, tempfile
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1 as S  # noqa: E402
from clinical_ir.relevance import Document  # noqa: E402
from clinical_ir.retrieval import BM25  # noqa: E402
from clinical_ir.evidence import EvidenceLevel as EL  # noqa: E402
from clinical_ir.temporal import temporal_decay  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'ok  ' if cond else 'FAIL'} {name}{'' if cond else f'  ({detail})'}", flush=True)


# ---------- 1. the decision rule ----------
def stat(mean, lo, hi, p_holm=None):
    d = {"mean": mean, "ci95": [lo, hi], "p": 0.5}
    if p_holm is not None:
        d["p_holm"] = p_holm
    return d


GAIN = stat(0.08, 0.06, 0.10)          # M - B0 >= 0.05, lower bound > 0
SIG = stat(0.03, 0.01, 0.05, 0.001)    # Holm-significant and positive
NS = stat(0.01, -0.01, 0.03, 0.30)     # not significant
NEG = stat(-0.03, -0.05, -0.01, 0.001) # significant, wrong sign
P_OK, N_OK = stat(0.05, 0.03, 0.07), stat(-0.03, -0.05, -0.01)


def verdict(primary=GAIN, P=P_OK, N=N_OK, manip=True, **over):
    sec = {k: SIG for k in ("M-F", "M-U", "M-V", "M-L", "M-Rbar")}
    sec.update({k.replace("_", "-"): v for k, v in over.items()})
    return S.decide(primary, sec, {"Nbar-B0": N, "Pbar-M": P}, manip)


def check_decision_rule():
    cases = [
        ("every condition met", {}, "Supported"),
        ("P lower bound 0 -> Test failed", dict(P=stat(0.02, 0.0, 0.04)), "Test failed"),
        ("P lower bound < 0 -> Test failed", dict(P=stat(0.01, -0.01, 0.03)), "Test failed"),
        ("Test failed outranks Refuted", dict(P=stat(0.0, -0.01, 0.01), primary=stat(0.0, -0.02, 0.02)), "Test failed"),
        ("Test failed outranks Untestable", dict(P=stat(0.0, -0.01, 0.01), manip=False), "Test failed"),
        ("M - B0 upper bound < 0.05 -> Refuted", dict(primary=stat(0.02, 0.0, 0.049)), "Refuted"),
        ("negative M - B0 -> Refuted", dict(primary=stat(-0.05, -0.08, -0.02)), "Refuted"),
        ("Refuted outranks Untestable (B2)", dict(primary=stat(0.02, 0.0, 0.04), manip=False), "Refuted"),
        ("manipulation check failed -> Untestable", dict(manip=False), "Untestable"),
        ("Untestable even with a null M - U", dict(manip=False, M_U=NS), "Untestable"),
        ("M - U not significant -> Gain without attribution", dict(M_U=NS), "Gain without attribution"),
        ("M - U significant but negative -> Gain without attribution", dict(M_U=NEG), "Gain without attribution"),
        ("M - V not significant -> Gain without attribution (A8)", dict(M_V=NS), "Gain without attribution"),
        ("N upper bound = 0.05 -> Gain without attribution", dict(N=stat(0.0, -0.05, 0.05)), "Gain without attribution"),
        ("N upper bound > 0.05 -> Gain without attribution", dict(N=stat(0.04, 0.02, 0.06)), "Gain without attribution"),
        ("M - F not significant -> Inconclusive", dict(M_F=NS), "Inconclusive"),
        ("M - F negative -> Inconclusive", dict(M_F=NEG), "Inconclusive"),
        ("M - L and M - Rbar do not enter the verdict", dict(M_L=NEG, M_Rbar=NS), "Supported"),
        ("mean >= 0.05 but lower bound <= 0 -> Inconclusive", dict(primary=stat(0.06, -0.01, 0.13)), "Inconclusive"),
        ("lower bound > 0, mean < 0.05, upper >= 0.05 -> Inconclusive", dict(primary=stat(0.04, 0.01, 0.07)), "Inconclusive"),
        ("upper bound exactly 0.05 is not Refuted", dict(primary=stat(0.03, 0.01, 0.05)), "Inconclusive"),
        ("mean exactly 0.05 counts as a gain", dict(primary=stat(0.05, 0.01, 0.09)), "Supported"),
        ("p_holm exactly 0.05 is not significant", dict(M_U=stat(0.03, 0.01, 0.05, 0.05)), "Gain without attribution"),
    ]
    reached = set()
    for name, kw, want in cases:
        got = verdict(**kw)
        reached.add(got)
        check(f"decide: {name}", got == want, f"got {got!r}, want {want!r}")
    every = {"Test failed", "Refuted", "Untestable", "Supported", "Gain without attribution", "Inconclusive"}
    check("decide: every verdict row is reached", reached == every, f"missing {every - reached}")


# ---------- 2. synthetic result sets ----------
def synthetic(kind, n=90, seed=1):
    rng = np.random.default_rng(seed)
    e = lambda sd=0.03: rng.normal(0, sd, n)
    B0 = 0.30 + e(0.10)
    A = {"B0": B0}
    if kind == "good":
        A["M"] = B0 + 0.10 + e()
        A.update(F=A["M"] - 0.10 + e(), U=A["M"] - 0.05 + e(), V=A["M"] - 0.05 + e(), L=A["M"] + e())
        A.update({f"N{s}": B0 - 0.02 + e() for s in S.SEEDS})
        A.update({f"R{s}": A["M"] - 0.03 + e() for s in S.SEEDS})
        A.update({f"P{s}": A["M"] + 0.10 + e() for s in S.SEEDS})
    else:  # every prediction fails, but the harness can still detect a real module
        A["M"] = B0 + e()
        A.update(F=A["M"] + 0.05 + e(), U=A["M"] + e(), V=A["M"] + e(), L=A["M"] + 0.05 + e())
        A.update({f"N{s}": B0 + 0.08 + e() for s in S.SEEDS})
        A.update({f"R{s}": A["M"] + e() for s in S.SEEDS})
        A.update({f"P{s}": A["M"] + 0.10 + e() for s in S.SEEDS})
    return A


def check_synthetic():
    good = S.analyze(synthetic("good"), True)
    check("synthetic: every prediction holds -> Supported", good["verdict"] == "Supported", good["verdict"])
    check("synthetic: Supported carries no qualifier", good["qualifiers"] == [], good["qualifiers"])
    check("synthetic: the Holm family has the five A8 comparisons",
          list(good["secondary_holm"]) == ["M-F", "M-U", "M-V", "M-L", "M-Rbar"], list(good["secondary_holm"]))
    bad = S.analyze(synthetic("bad"), True)
    check("synthetic: every prediction fails -> Refuted", bad["verdict"] == "Refuted", bad["verdict"])
    check("synthetic: both qualifiers fire when L beats M and Rbar ties M", len(bad["qualifiers"]) == 2, bad["qualifiers"])
    A = synthetic("good")
    A.update({f"P{s}": A["M"] + np.random.default_rng(s).normal(0, 0.03, 90) for s in S.SEEDS})
    tf = S.analyze(A, True)["verdict"]
    check("synthetic: P no better than M -> Test failed", tf == "Test failed", tf)
    un = S.analyze(synthetic("good"), False)["verdict"]
    check("synthetic: manipulation check failed -> Untestable", un == "Untestable", un)

    p = S.paired(np.full(20, 0.07))
    check("paired: a constant difference has a degenerate interval", np.allclose(p["ci95"], [0.07, 0.07]), p["ci95"])
    p0 = S.paired(np.zeros(20))
    check("paired: zero differences give p = 1", p0["p"] == 1.0, p0["p"])
    check("paired: seeded, so repeatable", S.paired(synthetic("good")["M"] - synthetic("good")["B0"]) ==
          S.paired(synthetic("good")["M"] - synthetic("good")["B0"]))
    h = S.holm([0.01, 0.04, 0.03])
    check("holm: step-down adjustment", np.allclose(h, [0.03, 0.06, 0.06]), h)


# ---------- 3. parsers ----------
NXML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD Journal Archiving and Interchange DTD v3.0 20080202//EN" "archivearticle3.dtd">
<article xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:mml="http://www.w3.org/1998/Math/MathML" article-type="research-article">
<front><journal-meta><journal-title>J</journal-title></journal-meta>
<article-meta>
<article-id pub-id-type="pmid">{pmid}</article-id>
<article-id pub-id-type="pmc">{pmc}</article-id>
<article-id pub-id-type="doi">10.1000/x{pmc}</article-id>
<title-group><article-title>{title}</article-title></title-group>
{dates}
{abstract}
</article-meta></front>
<body><p>BODYTEXT should never be read.</p></body></article>
"""


def nxml(pmc, title, abstract, dates, pmid=""):
    ab = f"<abstract>{abstract}</abstract>" if abstract else ""
    return NXML.format(pmc=pmc, pmid=pmid, title=title, dates=dates, abstract=ab)


def check_parsers():
    d = S.parse_nxml(nxml("3000001", "Aspirin&nbsp;and <italic>stroke</italic>\n prevention &amp; care",
                          "<p>Background text.</p><sec><title>Results</title><p>We recommend a low dose.</p></sec>"
                          "</abstract><abstract abstract-type=\"graphical\"><p>GRAPHICAL</p>",
                          '<pub-date pub-type="ppub"><month>3</month><year>2012</year></pub-date>'
                          '<pub-date pub-type="epub"><day>15</day><month>01</month><year>2012</year></pub-date>',
                          pmid="12345678"), "3000001")
    check("nxml: id, PMID and article type", (d["_id"], d["pmid"], d["article_type"]) == ("3000001", "12345678", "research-article"), d)
    check("nxml: title with markup and entities", d["title"] == "Aspirin and stroke prevention & care", d["title"])
    check("nxml: abstract sections joined, graphical abstract and body left out",
          d["abstract"] == "Background text. Results We recommend a low dose.", d["abstract"])
    check("nxml: epub preferred over ppub (B5)", d["pub_date"] == "2012-01-15", d["pub_date"])

    d = S.parse_nxml(nxml("PMC3000002", "T", "<p>A</p>", '<pub-date pub-type="ppub"><month>2</month><year>2010</year></pub-date>'),
                     "PMC3000002")
    check("nxml: PMC prefix stripped; no PMID is None", (d["_id"], d["pmid"]) == ("3000002", None), d)
    check("nxml: missing day becomes 1 (B5)", d["pub_date"] == "2010-02-01", d["pub_date"])
    d = S.parse_nxml(nxml("3000003", "T", "", '<pub-date date-type="pub" publication-format="electronic"><year>2015</year>'
                          '<month>Sep</month></pub-date>'), "3000003")
    check("nxml: no epub or ppub falls back to any date; non-numeric month becomes 1",
          d["pub_date"] == "2015-01-01", d["pub_date"])
    check("nxml: no abstract gives an empty abstract (title alone is used)", d["abstract"] == "", d["abstract"])
    d = S.parse_nxml(nxml("3000004", "T", "<p>A</p>", '<pub-date pub-type="epub"><day>31</day><month>2</month><year>2011</year></pub-date>'),
                     "3000004")
    check("nxml: an impossible day falls back to the first of the month", d["pub_date"] == "2011-02-01", d["pub_date"])
    d = S.parse_nxml(nxml("3000005", "T", "<p>A</p>", ""), "3000005")
    check("nxml: no pub-date gives None", d["pub_date"] is None, d["pub_date"])
    d = S.parse_nxml(nxml("PMC3000006", "T", "<p>A</p>", ""), "article")
    check("nxml: a non-PMCID file name falls back to the article-id", d["_id"] == "3000006", d["_id"])
    d = S.parse_nxml(nxml("3000099", "T", "<p>A</p>", ""), "3000007")
    check("nxml: the file name is the document id (TREC CDS), the article-id is recorded",
          (d["_id"], d["pmc_in_file"]) == ("3000007", "3000099"), d)
    bomb = ('<?xml version="1.0"?><!DOCTYPE article [<!ENTITY a "AAAAAAAAAA"><!ENTITY b "&a;&a;&a;&a;&a;">]>'
            '<article><front><article-meta><article-id pub-id-type="pmc">1</article-id>'
            '<title-group><article-title>x &b; y</article-title></title-group></article-meta></front></article>')
    d = S.parse_nxml(bomb, "1")
    check("nxml: declared entities are never expanded", d["title"] == "x y", d["title"])
    check("nxml: no article-meta gives None", S.parse_nxml("<article><front/></article>", "9") is None)

    t14 = """<topics>
  <topic number="1" type="diagnosis">
    <description>A 58-year-old woman with hypertension presents with chest pain.</description>
    <summary>58-year-old woman with
       hypertension and chest pain.</summary>
  </topic>
  <topic number="2" type="Test "><description>D</description><summary>Patient needs a test.</summary></topic>
  <topic number="3" type="treatment"><description>D</description><summary>Treat the sepsis.</summary></topic>
</topics>"""
    ts = S.parse_topics(t14, 2014)
    check("topics 2014: ids, types and columns (A4)",
          [(t["qid"], t["type"], t["column"]) for t in ts] ==
          [("2014-1", "diagnosis", "diagnosis_test"), ("2014-2", "test", "diagnosis_test"), ("2014-3", "treatment", "treatment")],
          [(t["qid"], t["type"], t["column"]) for t in ts])
    check("topics 2014: the summary is the query, whitespace collapsed",
          ts[0]["summary"] == "58-year-old woman with hypertension and chest pain.", ts[0]["summary"])
    t16 = """<?xml version="1.0" encoding="UTF-8"?><topics>
<topic number="1" type="diagnosis"><note>Long admission note with &lt;tags&gt;.</note>
<description>Description.</description><summary>Short summary.</summary></topic></topics>"""
    ts = S.parse_topics(t16, "2016")
    check("topics 2016: note and description are not used", (ts[0]["qid"], ts[0]["summary"]) == ("2016-1", "Short summary."), ts)
    try:
        S.parse_topics('<topics><topic number="4" type="test"><description>D</description></topic></topics>', 2015)
        check("topics: a missing summary is refused", False, "no error")
    except ValueError:
        check("topics: a missing summary is refused", True)
    ts = S.parse_topics('<topics><topic number="5" type="prognosis"><summary>S</summary></topic></topics>', 2015)
    check("topics: a type outside A4 has no column", ts[0]["column"] is None, ts)

    q = S.parse_qrels("1 0 3000001 2\n1 0 PMC3000002 0\n\n2 0 3000003 1\n", 2014)
    check("qrels: four-column TREC format, PMC prefix stripped",
          q == {"2014-1": {"3000001": 2.0, "3000002": 0.0}, "2014-2": {"3000003": 1.0}}, q)
    for bad in ("1 0 3000001 2\nnot a qrel line\n", "1 0 3000001 2\n10 0 3000004 7 1\n"):
        try:
            S.parse_qrels(bad, 2016)
            check(f"qrels: refused {bad.splitlines()[1]!r}", False, "no error")
        except ValueError:
            check(f"qrels: refused {bad.splitlines()[1]!r}", True)

    ef = """<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
<PubmedArticleSet>
<PubmedArticle><MedlineCitation Status="MEDLINE" Owner="NLM"><PMID Version="1">12345678</PMID>
<Article><ArticleTitle>T &#8211; x</ArticleTitle><PublicationTypeList>
<PublicationType UI="D016428">Journal Article</PublicationType>
<PublicationType UI="D016449">Randomized Controlled Trial</PublicationType></PublicationTypeList></Article>
<CommentsCorrectionsList><CommentsCorrections RefType="CommentIn"><PMID Version="1">999</PMID></CommentsCorrections></CommentsCorrectionsList>
</MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID Version="1">222</PMID><Article><PublicationTypeList>
<PublicationType UI="D016454">Review</PublicationType></PublicationTypeList></Article></MedlineCitation></PubmedArticle>
<PubmedBookArticle><BookDocument><PMID Version="1">333</PMID></BookDocument></PubmedBookArticle>
</PubmedArticleSet>"""
    got = S.parse_efetch(ef)
    check("efetch: publication types by PMID, nested PMIDs ignored",
          got == {"12345678": ["Journal Article", "Randomized Controlled Trial"], "222": ["Review"]}, got)


# ---------- 4. A5 mapping ----------
def check_mapping():
    cases = [(None, None), ([], None), (["Journal Article"], None), (["Review"], None),
             (["Systematic Review"], EL.SYSTEMATIC_REVIEW), (["Meta-Analysis", "Journal Article"], EL.SYSTEMATIC_REVIEW),
             (["Randomized Controlled Trial"], EL.RCT), (["Clinical Trial", "Randomized Controlled Trial"], EL.RCT),
             (["Controlled Clinical Trial"], EL.CONTROLLED_OBSERVATIONAL), (["Clinical Trial"], EL.CONTROLLED_OBSERVATIONAL),
             (["Observational Study"], EL.CONTROLLED_OBSERVATIONAL), (["Comparative Study"], EL.CONTROLLED_OBSERVATIONAL),
             (["Case Reports"], EL.UNCONTROLLED_OBSERVATIONAL), (["Letter", "Case Reports"], EL.UNCONTROLLED_OBSERVATIONAL),
             (["Editorial"], EL.EXPERT_OPINION), (["Comment"], EL.EXPERT_OPINION), (["Letter"], EL.EXPERT_OPINION),
             (["Consensus Development Conference"], EL.EXPERT_OPINION),
             (["Meta-Analysis", "Randomized Controlled Trial", "Editorial"], EL.SYSTEMATIC_REVIEW)]
    bad = [(c, S.pt_to_level(c), e) for c, e in cases if S.pt_to_level(c) != e]
    check(f"A5 mapping: {len(cases)} hand-written cases, the highest level applies", not bad, bad)


# ---------- 5. sparse BM25 ----------
def check_bm25():
    rng = np.random.default_rng(7)
    words = [f"w{i}" for i in range(40)] + ["Fever", "fever's", "x-ray", "β-blocker", "covid_19"]
    texts = []
    for i in range(300):
        if i % 37 == 5:
            texts.append(texts[i - 3])  # exact duplicates: tied scores
        elif i % 53 == 0:
            texts.append(" ")           # empty document
        else:
            texts.append(" ".join(rng.choice(words, rng.integers(3, 40))) + ", (end).")
    titles = [t.split(" ")[0] for t in texts]
    docs = [Document(id=str(i), title=titles[i], abstract=" ".join(texts[i].split(" ")[1:]),
                     content_embedding=np.zeros(1), publication_date=datetime(2020, 1, 1)) for i in range(300)]
    repo = BM25(); repo.index(docs)
    sp = S.SparseBM25(f"{d.title} {d.abstract}" for d in docs)  # a generator: read once, as in the run
    queries = [" ".join(rng.choice(words, rng.integers(1, 6))) for _ in range(40)]
    queries += ["w1 w1 w1", "nothing matches here", "FEVER x ray", "", "w3, w4; w5!"]
    mism, worst = 0, 0.0
    for q in queries:
        a = repo.search(q, S.POOL)
        b, s = sp.search(q, S.POOL)
        mism += [i for i, _ in a] != b
        worst = max(worst, max(abs(sc - s[i]) for i, sc in a))
    check(f"sparse BM25: top 100 identical to the repository's on {len(queries)} queries, ties included", mism == 0, mism)
    check("sparse BM25: scores identical to the repository's", worst == 0.0, worst)
    check("sparse BM25: document count from a generator", sp.N == 300, sp.N)


# ---------- 6. manipulation check and helpers ----------
def check_helpers():
    topics = [{"qid": f"q{i}", "column": "diagnosis_test" if i < 4 else "treatment"} for i in range(6)]
    base = {f"q{i}": list("abcdefghij") for i in range(6)}
    swap = list("bacdefghij")
    def run(n_differ, treat_differ=True):
        U = {q: list(v) for q, v in base.items()}
        M = {q: (swap if (int(q[1:]) < n_differ or (treat_differ and int(q[1:]) >= 4)) else list(v)) for q, v in base.items()}
        return S.manipulation_check(topics, {"M": M, "U": U}, {f"q{i}": i % 2 == 0 for i in range(6)})
    r = run(2)
    check("manipulation: 2 of 4 differ -> share 0.5, passes (>= 0.5)", (r["share_M_top10_differs_from_U"], r["passed"]) == (0.5, True), r)
    r = run(1)
    check("manipulation: 1 of 4 differ -> fails; treatment topics ignored", (r["share_M_top10_differs_from_U"], r["passed"]) == (0.25, False), r)
    check("manipulation: a reordering counts as different (ordered lists)", run(4)["share_M_top10_differs_from_U"] == 1.0)
    check("manipulation: evidence-varies share over diagnosis/test topics", run(2)["share_evidence_varies_in_pool"] == 0.5)
    r = S.manipulation_check([t for t in topics if t["column"] == "treatment"], {"M": base, "U": base}, {})
    check("manipulation: no diagnosis/test topics -> fails", r["passed"] is False, r)

    check("fill_pool_mean: missing values get the pool mean (B5)", np.allclose(S.fill_pool_mean([1, np.nan, 3]), [1, 2, 3]))
    check("fill_pool_mean: all missing gives zeros", np.allclose(S.fill_pool_mean([np.nan] * 3), 0))
    check("z: a constant module contributes 0", np.allclose(S.z([0.4] * 5), 0))
    check("rank_by: ties keep the pool's BM25 order", S.rank_by([1, 2, 2, 1], list("abcd")) == list("bcad"))
    Z = np.arange(18, dtype=float).reshape(6, 3)
    F = S.l_features(Z, "diagnosis_test")
    col = S.TYPES.index("diagnosis_test")
    check("l_features: 6 z-scores plus 6 x 4 interactions; only the topic's column is filled",
          F.shape == (3, 30) and np.allclose(F[:, 6 + 6 * col:12 + 6 * col], Z.T)
          and np.allclose(np.delete(F[:, 6:], np.s_[6 * col:6 * col + 6], axis=1), 0), F.shape)
    g = S.gated(Z, S.GATES["treatment"], {"applicability"})
    check("gated: an off module contributes nothing", np.allclose(g, Z[[0, 1, 2, 3, 5]].sum(0)))
    check("temporal: a document with no date gets tau = 1 in F (B5)", temporal_decay(datetime(2014, 1, 21), datetime(2014, 1, 21)) == 1.0)
    check("L grid (B1)", S.LAMBDAS == [0.001, 0.01, 0.1, 1, 10], S.LAMBDAS)
    with tempfile.TemporaryDirectory() as d:
        enc = S.Encoder(d, fake=True)
        a1, a2 = enc(["one", "two"], "medcpt_query"), enc(["one", "two"], "medcpt_query")
        b = enc(["one", "three"], "medcpt_query")
        check("encoder cache: keyed by the texts, not their count", np.array_equal(a1, a2) and len(os.listdir(d)) == 2
              and not np.array_equal(a1[1], b[1]))


# ---------- 7. end to end ----------
VOCAB = ("aspirin stroke anemia transfusion sepsis pneumonia fever cough biopsy mri ultrasound lesion cardiac renal "
         "hepatic infant elderly pain rash fracture insulin glucose tumor lymphoma asthma dyspnea syncope seizure").split()
KEYS = ("kawasaki sarcoidosis endocarditis amyloidosis pheochromocytoma myasthenia histoplasmosis porphyria "
        "takayasu wilson brucellosis tularemia whipple behcet addison cushing hemochromatosis leptospirosis").split()
ACTION = ["we recommend", "the guideline", "dose", "protocol"]
PUBTYPES = [["Journal Article"], ["Journal Article", "Randomized Controlled Trial"], ["Meta-Analysis"],
            ["Case Reports"], ["Editorial"], None]


def make_collection(rng, n, base_id, dated_share, years):
    docs = []
    for i in range(n):
        keys = [k for k in KEYS if rng.random() < 0.12]
        title = " ".join(list(rng.choice(VOCAB, 3)) + keys[:1])
        abstract = " ".join(list(rng.choice(VOCAB, rng.integers(15, 50))) + keys + (["we recommend a dose"] if rng.random() < 0.3 else []))
        date = None
        if rng.random() < dated_share:
            date = f'<pub-date pub-type="epub"><day>{rng.integers(1, 28)}</day><month>{rng.integers(1, 13)}</month><year>{rng.integers(*years)}</year></pub-date>'
        pmid = str(20000000 + base_id + i) if rng.random() < 0.85 else ""
        docs.append({"pmc": str(base_id + i), "title": title, "abstract": abstract, "date": date or "", "pmid": pmid, "keys": keys})
    return docs


def write_fixture(root):
    rng = np.random.default_rng(3)
    c14 = make_collection(rng, 170, 3000000, 1.0, (1995, 2014))
    c14[0]["date"] = '<pub-date pub-type="epub"><year>2015</year></pub-date>'  # after the reference date: tau clamps to 1
    c16 = make_collection(rng, 160, 4000000, 0.75, (1995, 2016))
    with tarfile.open(os.path.join(root, "c2014.tar.gz"), "w:gz") as tf:
        def add(name, text):
            p = os.path.join(root, "member.tmp")
            with open(p, "w") as f:
                f.write(text)
            tf.add(p, arcname=name)
        for d in c14:
            add(f"pmc-text-00/{d['pmc'][-2:]}/{d['pmc']}.nxml", nxml(d["pmc"], d["title"] + " &nbsp;&amp;", "<p>" + d["abstract"] + "</p>", d["date"], d["pmid"]))
        add("pmc-text-01/00/3000005.nxml", nxml("3000005", "duplicate", "<p>x</p>", ""))   # duplicate id
        add("pmc-text-01/00/3999999.nxml", "<article><front><article-meta>")              # malformed
        os.remove(os.path.join(root, "member.tmp"))
    os.makedirs(os.path.join(root, "c2016_nxml", "sub"))
    for d in c16:
        with open(os.path.join(root, "c2016_nxml", "sub", f"PMC{d['pmc']}.nxml"), "w") as f:
            f.write(nxml("PMC" + d["pmc"], d["title"], "<p>" + d["abstract"] + "</p>", d["date"], d["pmid"]))
    for name, coll in (("c2014", c14), ("c2016", c16)):
        p = os.path.join(root, f"{name}_pubtypes.jsonl")
        with open(p, "w") as f:
            for d in coll:
                if d["pmid"]:
                    f.write(json.dumps({"pmid": d["pmid"], "types": PUBTYPES[rng.integers(len(PUBTYPES))]}) + "\n")
        S.write_meta(p, {"step": "fixture"})

    types = ["diagnosis", "test", "treatment"] * 2
    topic_keys = iter(KEYS)
    for year, coll, layout in (("2014", c14, "2014"), ("2015", c14, "2014"), ("2016", c16, "2016")):
        topics, qrels = [], []
        for n, kind in enumerate(types, start=1):
            key = next(topic_keys)
            summary = f"Patient with {' '.join(rng.choice(VOCAB, 4))} and suspected {key}."
            note = "<note>Admission note.</note>" if layout == "2016" else ""
            topics.append(f'<topic number="{n}" type="{kind}">{note}<description>Long description.</description>'
                          f'<summary>{summary}</summary></topic>')
            for d in coll:
                if key in d["keys"]:
                    qrels.append(f"{n} 0 {d['pmc']} {2 if key in d['title'] else 1}")
            for d in rng.choice(coll, 20, replace=False):
                if key not in d["keys"]:
                    qrels.append(f"{n} 0 {d['pmc']} 0")
        with open(os.path.join(root, f"topics{year}.xml"), "w") as f:
            f.write("<topics>\n" + "\n".join(topics) + "\n</topics>\n")
        with open(os.path.join(root, f"qrels{year}.txt"), "w") as f:
            f.write("\n".join(qrels) + "\n")

    nf = os.path.join(root, "nfcorpus"); os.makedirs(os.path.join(nf, "qrels"))
    with open(os.path.join(nf, "corpus.jsonl"), "w") as f:
        for i in range(120):
            f.write(json.dumps({"_id": f"MED-{i}", "title": " ".join(rng.choice(VOCAB, 3)),
                                "text": " ".join(rng.choice(VOCAB, 30))}) + "\n")
    with open(os.path.join(nf, "queries.jsonl"), "w") as f:
        for i in range(10):
            f.write(json.dumps({"_id": f"PLAIN-{i}", "text": " ".join(rng.choice(VOCAB, 3))}) + "\n")
    with open(os.path.join(nf, "qrels", "test.tsv"), "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for i in range(10):
            for j in rng.choice(120, 5, replace=False):
                f.write(f"PLAIN-{i}\tMED-{j}\t{rng.integers(1, 3)}\n")

    cfg = {"nfcorpus": "nfcorpus",
           "years": {y: {"topics": f"topics{y}.xml", "qrels": f"qrels{y}.txt", "collection": c}
                     for y, c in (("2014", "c2014"), ("2015", "c2014"), ("2016", "c2016"))},
           "collections": {"c2014": {"corpus": "c2014.jsonl", "pubtypes": "c2014_pubtypes.jsonl", "reference_date": "2014-01-21"},
                           "c2016": {"corpus": "c2016.jsonl", "pubtypes": "c2016_pubtypes.jsonl", "reference_date": "2016-03-28"}}}
    return cfg


def cli(*args):
    return subprocess.run([sys.executable, os.path.join(HERE, "stage1.py"), *args], capture_output=True, text=True)


def write_cfg(root, cfg, name):
    p = os.path.join(root, name)
    with open(p, "w") as f:
        json.dump(cfg, f)
    return p


def check_end_to_end():
    with tempfile.TemporaryDirectory() as root:
        cfg = write_fixture(root)
        r1 = cli("prepare-corpus", os.path.join(root, "c2014.tar.gz"), "--out", os.path.join(root, "c2014.jsonl"))
        r2 = cli("prepare-corpus", os.path.join(root, "c2016_nxml"), "--out", os.path.join(root, "c2016.jsonl"))
        check("prepare-corpus: tar.gz and directory sources", r1.returncode == 0 and r2.returncode == 0, r1.stderr[-800:] + r2.stderr[-800:])
        m = json.load(open(os.path.join(root, "c2014.jsonl.meta.json")))
        check("prepare-corpus: counts written, the duplicate and the malformed file set aside",
              m["counts"].get("written") == 170 and m["counts"].get("duplicate_id") == 1 and m["counts"].get("parse_error") == 1, m["counts"])
        check("prepare-corpus: checksum and input checksums recorded",
              m["sha256"] == S.sha256_file(os.path.join(root, "c2014.jsonl")) and len(m["inputs"]) == 1, m)
        first = json.loads(open(os.path.join(root, "c2014.jsonl")).readline())
        check("prepare-corpus: entities in titles handled", first["title"].endswith("&"), first["title"])
        m16 = json.load(open(os.path.join(root, "c2016.jsonl.meta.json")))["counts"]
        check("prepare-corpus: 2016 date coverage below 95%", m16["with_date"] / m16["written"] < 0.95, m16)

        config = write_cfg(root, cfg, "config.json")
        cache = os.path.join(root, "cache")
        out1, out2 = os.path.join(root, "out1"), os.path.join(root, "out2")
        r = cli("run", "--config", config, "--out", out1, "--cache", cache, "--test-mode", "--fake-encoder")
        check("run: test mode with the stand-in encoder completes", r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:])
        if r.returncode != 0:
            return
        R = json.load(open(os.path.join(out1, "results.json")))
        arms = ["B0", "F", "M", "U", "V", "L"] + [f"{a}{s}" for a in "NRP" for s in S.SEEDS]
        check("run: a run file for every arm", sorted(os.listdir(os.path.join(out1, "runs"))) == sorted(f"{a}.trec" for a in arms),
              os.listdir(os.path.join(out1, "runs")))
        lines = open(os.path.join(out1, "runs", "M.trec")).read().splitlines()
        per_topic = {}
        for l in lines:
            q, _, d, rank, _, _ = l.split()
            per_topic.setdefault(q, []).append(d)
        check("run: 18 topics, 100 distinct documents each, in TREC format",
              len(per_topic) == 18 and all(len(v) == 100 == len(set(v)) for v in per_topic.values()), {k: len(v) for k, v in per_topic.items()})
        check("run: verdict is one of the registered rows",
              R["verdict"] in ("Test failed", "Refuted", "Untestable", "Supported", "Gain without attribution", "Inconclusive"), R["verdict"])
        check("run: in-run fidelity checks pass",
              R["fidelity"]["checks"] == {"M_bm25_only_equals_B0": "18/18", "M_equals_U_on_treatment": "6/6", "F_formula_mismatch": 0},
              R["fidelity"]["checks"])
        check("run: A6 gate on the fixture NFCorpus (no target in test mode)",
              R["fidelity"]["a6_sparse_bm25_gate"]["passed"] and R["fidelity"]["a6_sparse_bm25_gate"]["top100_mismatches"] == 0)
        dc = R["preconditions"]["date_coverage"]
        check("run: temporal on for full date coverage, off below 95% (B5)",
              dc["c2014"]["temporal_on"] and not dc["c2016"]["temporal_on"], dc)
        check("run: topics by column", R["topics"]["by_column"] == {"diagnosis_test": 12, "treatment": 6}, R["topics"])
        ev = R["events"]
        idx = lambda s: next(i for i, e in enumerate(ev) if s in e)
        check("run: B3 order -- rankings saved, then the manipulation check, then judgments read",
              idx("Judgment-free rankings saved") < idx("Manipulation check saved") < idx("Judgments read"), ev)
        mc = json.load(open(os.path.join(out1, "manipulation_check.json")))
        check("run: manipulation check saved with the registered threshold", mc["threshold"] == 0.5 and mc["diagnosis_test_topics"] == 12, mc)
        check("run: L cross-fitted in 5 outer folds on the B1 grid",
              len(R["L"]["folds"]) == 5 and all(set(f["cv"]) == {str(l) for l in S.LAMBDAS} for f in R["L"]["folds"]), R["L"])
        check("run: P's noise is half the judgment range (B5)", R["p_noise_sd"] == 1.0, R["p_noise_sd"])
        rows = list(csv.reader(open(os.path.join(out1, "per_query.csv"))))
        check("run: per-query values for every arm", len(rows) == 19 and set(arms) <= set(rows[0]), rows[0])
        man = json.load(open(os.path.join(out1, "manifest.json")))
        want = {"c2014/corpus", "c2014/pubtypes", "c2016/corpus", "c2016/pubtypes", "nfcorpus/corpus.jsonl",
                "nfcorpus/queries.jsonl", "nfcorpus/qrels/test.tsv"} | {f"{y}/{k}" for y in ("2014", "2015", "2016") for k in ("topics", "qrels")}
        check("run: manifest has every data checksum, the script checksum and the stand-in flag",
              set(man["data_sha256"]) == want and man["fake_encoder"] is True
              and man["script_sha256"] == S.sha256_file(os.path.join(HERE, "stage1.py")), set(man["data_sha256"]) ^ want)
        report = open(os.path.join(out1, "RESULTS.md")).read()
        check("run: generated report is marked as test mode", "TEST MODE" in report and R["verdict"] in report)

        r = cli("run", "--config", config, "--out", out2, "--cache", cache, "--test-mode", "--fake-encoder")
        same_csv = r.returncode == 0 and open(os.path.join(out1, "per_query.csv")).read() == open(os.path.join(out2, "per_query.csv")).read()
        strip = lambda d: {k: v for k, v in d.items() if k not in ("events", "runtime_seconds")}
        same_json = r.returncode == 0 and strip(R) == strip(json.load(open(os.path.join(out2, "results.json"))))
        check("run: a second run gives identical per-query values and results", same_csv and same_json, r.stderr[-800:])

        r = cli("run", "--config", config, "--out", os.path.join(root, "o3"), "--cache", cache, "--fake-encoder")
        check("refusal: the stand-in encoder outside test mode", r.returncode != 0 and "stand-in encoder" in r.stderr, r.stderr[-400:])
        r = cli("run", "--config", config, "--out", os.path.join(root, "o4"), "--cache", cache)
        check("refusal: a real run holds the A6 gate to the registered NDCG@10 target",
              r.returncode != 0 and "A6 sparse BM25 fidelity gate failed" in r.stderr, r.stderr[-400:])

        tampered = os.path.join(root, "c2014_tampered.jsonl")  # one byte added, metadata copied unchanged
        with open(tampered, "w") as f:
            f.write(open(os.path.join(root, "c2014.jsonl")).read() + "\n")
        shutil.copy(os.path.join(root, "c2014.jsonl.meta.json"), tampered + ".meta.json")
        bad = json.loads(json.dumps(cfg)); bad["collections"]["c2014"]["corpus"] = "c2014_tampered.jsonl"
        r = cli("run", "--config", write_cfg(root, bad, "bad1.json"), "--out", os.path.join(root, "o5"), "--cache", cache,
                "--test-mode", "--fake-encoder")
        check("refusal: a corpus whose checksum does not match its metadata (B6)",
              r.returncode != 0 and "does not match" in r.stderr, r.stderr[-400:])
        bad = json.loads(json.dumps(cfg)); bad["collections"]["c2016"]["reference_date"] = ""
        r = cli("run", "--config", write_cfg(root, bad, "bad2.json"), "--out", os.path.join(root, "o6"), "--cache", cache,
                "--test-mode", "--fake-encoder")
        check("refusal: a missing reference date (B6)", r.returncode != 0 and "reference_date" in r.stderr, r.stderr[-400:])

        bad = json.loads(json.dumps(cfg))
        for y in bad["years"].values():
            y["qrels"] = "no_such_qrels.txt"
        o7 = os.path.join(root, "o7")
        r = cli("run", "--config", write_cfg(root, bad, "bad3.json"), "--out", o7, "--cache", cache, "--test-mode", "--fake-encoder")
        check("B3: with judgments unreadable, every judgment-free ranking and the manipulation check are still written",
              r.returncode != 0 and os.path.exists(os.path.join(o7, "manipulation_check.json"))
              and sorted(os.listdir(os.path.join(o7, "runs"))) == sorted(f"{a}.trec" for a in ["B0", "F", "M", "U", "V"] +
                                                                         [f"{x}{s}" for x in "NR" for s in S.SEEDS])
              and not os.path.exists(os.path.join(o7, "results.json")), r.stderr[-400:])
        check("B3: the judgment-free files do not depend on the judgments",
              open(os.path.join(o7, "manipulation_check.json")).read() == open(os.path.join(out1, "manipulation_check.json")).read()
              and open(os.path.join(o7, "runs", "M.trec")).read() == open(os.path.join(out1, "runs", "M.trec")).read())


def main():
    for part in (check_decision_rule, check_synthetic, check_parsers, check_mapping, check_bm25, check_helpers, check_end_to_end):
        print(f"\n# {part.__name__.removeprefix('check_').replace('_', ' ')}")
        part()
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

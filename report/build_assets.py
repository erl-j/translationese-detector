#!/usr/bin/env python3
"""Generate the figures, tables and example boxes of the technical report.

    ../.venv-hybrid/bin/python build_assets.py            # from report/ (needs matplotlib)
    python3 build_assets.py --refresh-examples            # re-extract the example texts from trn

Everything in report/assets/ is written by this script; the .qmd only \\input's it.
Sources (nothing is typed in here except the example selection and the analyst notes):

  tsw/rules/*.py (via tsw.engine.Engine)     rule metadata and hand-set values
  docs/INTEGRATION.md  sections 4 and 5      engine / baseline results, family ablation
  hybrid/reports/METRICS.md                  hybrid models on val
  hybrid/reports/stacking_{sent,100,500}.md  rule-alone AUC and learned multipliers (trn)
  data/MANIFEST.md                           split sizes
  report/examples_trn.json                   the trn example texts (ids from the balanced trn
                                             files; re-extract with --refresh-examples)

Example annotations are computed fresh with Engine().annotate() at build time, so the
highlighted spans and contributions always match the current rules. The analyst notes
below name the rules they rely on (`expect`); the build fails if one of them no longer
contributes, so a note cannot silently drift away from the engine's evidence.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(HERE, "assets")
sys.path.insert(0, ROOT)

from tsw.engine import Engine  # noqa: E402
from tsw.rules.base import ChoiceRule, HitRule, PredicateRule, RateRule, StatRule  # noqa: E402

EX_FILE = os.path.join(HERE, "examples_trn.json")

# ---------------------------------------------------------------------------
# Example selection (trn only) and analyst notes. `expect`: rules the note relies on,
# with the sign of their contribution (+ = toward translated).

EXAMPLES = [
    {"id": "ep-00-10-04:312:1:s34", "gran": "sent",
     "gloss": "Finally, I was both pleased and flattered when President Chirac referred to this "
              "report in his speech to Parliament a few weeks ago.",
     "expect": {"connpunct.initial_adverbial_comma": +1, "wordorder.v2_violation": +1,
                "connpunct.calque_connectives": +1},
     "note": r"\emph{Till sist, jag var} follows English \emph{Finally, I was}. Three rules read "
             r"three properties of the same construction: the comma after the fronted adverbial "
             r"(punctuation, at its rule cap of 2), the missing V2 inversion (word order) and the "
             r"structuring conjunct \emph{till sist} `finally' (lexis)."},
    {"id": "ep-99-05-07:34:1:s13", "gran": "sent",
     "gloss": "Personally, you see, I think that the campaign to prevent violence against women "
              "is actually more important for the future than this world exhibition.",
     "expect": {"particles.namligen": -1, "register.tycker_vs_anser": -1,
                "wordorder.fronted_theme": -1},
     "note": r"Three native choices without an English trigger: the clause-internal particle "
             r"\emph{nämligen} `you see' (at its rule cap of 2.5), \emph{tycker} rather than "
             r"\emph{anser} `think', and the fronted adverb with V2 inversion "
             r"\emph{Personligen tycker jag} `personally think I'."},
    {"id": "ep-01-10-03:313:1:s14", "gran": "sent",
     "gloss": "Finally, for this work to be successful, the Commission must not be left alone.",
     "expect": {"connpunct.initial_adverbial_comma": +1, "connpunct.calque_connectives": +1,
                "register.kommer_att_vs_ska": -1},
     "note": r"An original sentence that opens like Example~1, \emph{Till sist,} with a comma. "
             r"The comma rule (at its cap) and \emph{till sist} outweigh \emph{skall}, which "
             r"counts as the native side of the \emph{kommer att} vs \emph{ska/skall} choice. "
             r"Here the comma precedes a fronted purpose clause (\emph{för att} \ldots), where "
             r"it is ordinary Swedish punctuation: a false positive of the comma rule."},
    {"id": "ep-98-10-07:147:1:s3", "gran": "sent",
     "gloss": "Then we thought that it might go well in Hong Kong in purely economic terms, but "
              "that politically it would go very badly.",
     "expect": {"register.valdigt_vs_mycket": -1, "register.bra": -1,
                "wordorder.fronted_theme": -1},
     "note": r"A translation with three native-register choices: \emph{väldigt} rather than "
             r"\emph{mycket} before an adjective, evaluative \emph{bra} `good, well', and V2 "
             r"inversion after the fronted adverb (\emph{Då tänkte vi} `then thought we'). "
             r"Apart from small absence and length terms, no evidence points to translation."},
    {"id": "c100:1998-03-10:1:3", "gran": "100",
     "gloss": None,
     "expect": {"register.kommer_att_vs_ska": +1, "connpunct.calque_connectives": +1,
                "connpunct.initial_adverbial_comma": +1, "register.skulle_vilja_vs_vill": +1,
                "particles.poss_3rd": +1},
     "note": r"Evidence from several families: future \emph{kommer att} with no \emph{ska/skall} "
             r"(the choice rule at its cap of 2.5), \emph{Slutligen,} with a comma, the "
             r"structuring conjuncts \emph{i synnerhet} `in particular' and \emph{i allmänhet} "
             r"`in general', \emph{skulle jag vilja} `I would like to', and the possessive "
             r"\emph{hennes} `her'. The connectives-and-punctuation family exceeds its cap of 4 "
             r"and is clipped."},
    {"id": "c100:2002-04-09:0:27", "gran": "100",
     "gloss": None,
     "expect": {"register.kommer_att_vs_ska": +1, "calques.lexicon": +1,
                "stats.av_nominal": +1, "connpunct.som_drop": +1},
     "note": r"An original speech in which the speaker speaks for the Commission (\emph{det "
             r"förslag vi kommer att lägga fram} `the proposal we will put forward'). Future "
             r"\emph{kommer att}, \emph{säkerställa} `ensure' (calque lexicon), the nominal "
             r"\emph{betänkandet av} `the report by' and the relative clause without \emph{som} "
             r"all point to translation. Commission prose given in Swedish shares these "
             r"conventions with translated Commission prose (Section~\ref{sec-limitations})."},
]

# Rules shown in the main feature table (the rest are in the appendix), with a gloss
# written for the report (Swedish in \emph{}).
MAIN_RULES = [
    ("calques.lexicon", r"43 calques, e.g.\ \emph{det faktum att} `the fact that', \emph{när det rör} `as regards'"),
    ("calques.pied_piped_relative", r"\emph{i vilket}, \emph{för vilka} `in which, for which'"),
    ("connpunct.native_connectives", r"middle-field \emph{dessutom, därför, dock, därmed, inte minst}"),
    ("connpunct.additive_ocksa", r"additive \emph{också} `also'"),
    ("connpunct.calque_connectives", r"\emph{i synnerhet, i själva verket, till sist} `in particular, in fact, finally'"),
    ("connpunct.initial_och", r"sentence-initial \emph{Och} `And'"),
    ("connpunct.initial_adverbial_comma", r"comma after a fronted adverbial: \emph{Dessutom, vi \ldots}"),
    ("connpunct.clause_comma_omission", r"no comma before clausal \emph{men, eftersom, medan, vilket}"),
    ("connpunct.written_out_abbrev", r"\emph{till exempel} vs \emph{t.ex.}, \emph{så kallad} vs \emph{s.k.}"),
    ("starter.modal_particles", r"modal particles \emph{ju, nog, väl, visst}"),
    ("starter.sent_len", r"words per sentence"),
    ("particles.namligen", r"clause-internal \emph{nämligen} `you see, namely'"),
    ("particles.stance", r"\emph{egentligen, visserligen, i och för sig, i alla fall}"),
    ("particles.generic_man", r"generic \emph{man} `one'"),
    ("particles.ni_vs_man", r"\emph{ni} `you (pl.)' vs generic \emph{man}"),
    ("particles.poss_3rd", r"\emph{hans, hennes, deras, dess} `his, her, their, its'"),
    ("register.kommer_att_vs_ska", r"future \emph{kommer att} vs \emph{ska/skall}"),
    ("register.skulle_vilja_vs_vill", r"\emph{jag/vi skulle vilja} vs \emph{jag/vi vill} `would like to' vs `want to'"),
    ("register.tycker_vs_anser", r"opinion verb \emph{tycka} vs \emph{anse} `think, consider'"),
    ("register.valdigt_vs_mycket", r"\emph{väldigt} vs \emph{mycket} + adjective `very'"),
    ("register.bland_annat", r"\emph{bland annat, bl.a.} `among other things'"),
    ("register.samt", r"coordinator \emph{samt} `and, as well as'"),
    ("register.dem_har_rate", r"demonstrative \emph{den/det/de här} `this, these'"),
    ("stats.av_other", r"share of \emph{av} `of, by' outside nominalisations"),
    ("wordorder.fronted_theme", r"topicalised non-subject + V2: \emph{Därför anser jag}, \emph{Det vet vi}"),
    ("wordorder.subject_theme", r"subject in the prefield: pronoun, NP, \emph{det}, cleft"),
    ("wordorder.v2_violation", r"no inversion after a fronted adverbial, V3, embedded negation"),
]

FAMILY_ORDER = ["starter", "connpunct", "register", "particles", "calques", "wordorder", "stats"]
FAMILY_NAME = {"starter": "starter", "connpunct": "connectives \\& punctuation",
               "register": "register", "particles": "particles \\& pronouns",
               "calques": "calques", "wordorder": "word order", "stats": "statistics"}


# ---------------------------------------------------------------------------
# helpers

def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def write(name, text):
    os.makedirs(ASSETS, exist_ok=True)
    with open(os.path.join(ASSETS, name), "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote assets/%s" % name)


_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(s):
    s = "".join(_TEX.get(c, c) for c in s)
    s = s.replace("−", "$-$").replace("...", r"\ldots{}")
    s = re.sub(r"(?<![\w])'(?=[\w.(])", "`", s)     # opening single quotes
    s = s.replace('"', "''")
    return s


def md_rows(md, start_marker, stop=r"^\s*$"):
    """Rows (list of cells) of the first pipe table after the line containing start_marker."""
    lines = md.splitlines()
    i = next(k for k, ln in enumerate(lines) if start_marker in ln)
    rows, started = [], False
    for ln in lines[i + 1:]:
        if ln.startswith("|"):
            started = True
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
        elif started:
            break
    return rows[0], rows[1:]


def strip_md(s):
    return s.replace("**", "").replace("`", "").strip()


def accauc(cell):
    """'.653 / .723' or '0.612 / 0.612 / **0.666**' -> (acc, auc) as strings."""
    parts = [strip_md(p) for p in cell.split("/")]
    return parts[0], parts[-1]


def f3(x):
    """'0.666' -> '.666' (the docs' style)."""
    return x[1:] if x.startswith("0.") else x


# ---------------------------------------------------------------------------
# sources

def integration_results():
    """docs/INTEGRATION.md section 5: {(system, split): {col: (acc, auc)}}."""
    md = read("docs/INTEGRATION.md")
    sec = md[md.index("## 5. Results"):]
    head, rows = md_rows(sec, "acc / AUC, balanced")
    out = {}
    for r in rows:
        name, split = strip_md(r[0]), r[1]
        out[(name, split)] = {h: accauc(c) for h, c in zip(head[2:], r[2:])}
    return out


def integration_ablation():
    md = read("docs/INTEGRATION.md")
    sec = md[md.index("## 4. Leave-one-family-out"):]
    head, rows = md_rows(sec, "Exact from the per-text family sums")
    return head, rows


def hybrid_metrics():
    """hybrid/reports/METRICS.md: {gran: {model: {'what', 'val': (acc, auc), 'val_masked', 'cv'}}}."""
    md = read("hybrid/reports/METRICS.md")
    out = {}
    for gran in ("sent", "100", "500"):
        sec = md[md.index("## gran = %s" % gran):]
        head, rows = md_rows(sec, "## gran = %s" % gran)
        d = {}
        for r in rows:
            d[strip_md(r[0])] = {"what": r[1], "val": accauc(r[2]), "val_masked": accauc(r[3]),
                                 "cv": r[4] if r[4] != "-" else None}
        out[gran] = d
    return out


def stacking():
    """hybrid/reports/stacking_<gran>.md: {gran: {rule: {'alone_auc', 'mult'}}}."""
    out = {}
    for gran in ("sent", "100", "500"):
        md = read("hybrid/reports/stacking_%s.md" % gran)
        head, rows = md_rows(md, "LR over the engine")
        idx = {h: k for k, h in enumerate(head)}
        out[gran] = {strip_md(r[0]): {"alone_auc": r[idx["alone AUC"]],
                                      "mult": float(r[idx["multiplier"]]),
                                      "mult_s": r[idx["multiplier"]].lstrip("+")} for r in rows}
    return out


def manifest_counts():
    md = read("data/MANIFEST.md")
    head, rows = md_rows(md, "## Files")
    d = {}
    for r in rows:
        d[strip_md(r[0])] = {"n": int(r[1]), "o": int(r[2]), "t": int(r[3]),
                             "tok_o": int(r[4]), "tok_t": int(r[5])}
    m = re.search(r"Session dates present: trn (\S+) \.\. (\S+) \((\d+) sessions\); "
                  r"val (\S+) \.\. (\S+) \((\d+) sessions\); test (\S+) \.\. (\S+) \((\d+) sessions\)", md)
    dates = {"trn": m.group(1, 2, 3), "val": m.group(4, 5, 6), "test": m.group(7, 8, 9)}
    return d, dates


# ---------------------------------------------------------------------------
# tables

def rule_values(r):
    """Hand-set class parameters of a rule, as typed in tsw/rules (O / T)."""
    def g(x):
        return ("%.2g" % x).lstrip("0") if x < 1 else "%.2g" % x
    if isinstance(r, ChoiceRule):
        return r"P(A) %s / %s" % (g(r.p_o), g(r.p_t))
    if isinstance(r, RateRule):
        return r"%s / %s" % (g(r.rate_o), g(r.rate_t))
    if isinstance(r, StatRule):
        scale = getattr(r, "scale", 1.0) or 1.0
        unit = getattr(r, "unit_label", "")
        if scale != 1.0:
            return r"%s / %s" % (g(r.mu_o * scale), g(r.mu_t * scale))
        return r"%s / %s" % (g(r.mu_o), g(r.mu_t))
    if isinstance(r, HitRule):
        if r.per_label:
            v = sorted(r.per_label.values())
            return r"%+.1f to %+.1f per hit" % (v[0], v[-1])
        return r"%+.1f per hit" % r.per_hit
    return "per-sentence LLRs"


KIND = {ChoiceRule: "choice", RateRule: "rate", StatRule: "stat", HitRule: "hit",
        PredicateRule: "pred."}


def kind(r):
    for k, v in KIND.items():
        if isinstance(r, k):
            return v
    return "?"


def build_feature_table(eng, stack):
    rules = {r.name: r for r in eng.rules}
    rows = []
    fam = None
    for name, gloss in sorted(MAIN_RULES, key=lambda x: (FAMILY_ORDER.index(x[0].split(".")[0]),
                                                         MAIN_RULES.index(x))):
        r = rules[name]
        if r.family != fam:
            fam = r.family
            rows.append(r"\addlinespace[2pt]\multicolumn{6}{l}{\textit{%s} (family cap %g)}\\"
                        % (FAMILY_NAME[fam], eng.family_caps[fam]))
        s = stack["100"][name]
        rows.append(r"\texttt{%s} & %s & %s & %s & %s & %s \\"
                    % (tex(name.split(".", 1)[1]), r.direction, gloss, rule_values(r),
                       f3(s["alone_auc"]), s["mult_s"]))
    body = "\n".join(rows)
    write("tab_features.tex", r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Main rules by family. Dir: the direction a hit points to (O original, T translated).
Hand-set values are typed from trn statistics (O / T): rate rules in hits per 1000 words,
choice rules as the share of variant A (the first-named variant), stat rules as class means
(words per sentence; \emph{av} per 1000 words);
for the hit rules, log-likelihood ratio per hit. AUC: the rule's contribution alone as a score
on trn 100-token units; $\times$LR: multiplier on the hand-set contribution learned by a logistic
regression over all 49 rule contributions on trn (Section~\ref{sec-hybrid}).
Full list in Appendix~\ref{sec-appendix}.}
\label{tab-features}
\begin{tabularx}{\textwidth}{@{}l c X l r r@{}}
\toprule
rule & dir & feature & hand-set O / T & AUC & $\times$LR \\
\midrule
%s
\bottomrule
\end{tabularx}
\end{table}
""" % body)


def build_appendix(eng):
    rows = []
    for fam in FAMILY_ORDER:
        rs = [r for r in eng.rules if r.family == fam]
        rows.append(r"\addlinespace[3pt]\multicolumn{6}{@{}l}{\textit{%s} (%d rules, family cap %g)}\\*"
                    % (FAMILY_NAME[fam], len(rs), eng.family_caps[fam]))
        for r in rs:
            rows.append(r"\texttt{%s} & %s & %s & %g & %.2f & %s \\"
                        % (tex(r.name.split(".", 1)[1]), kind(r), r.direction, r.cap, r.weight,
                           tex(r.description)))
    write("tab_rules_appendix.tex", r"""{\footnotesize
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.21\textwidth} l c r r >{\raggedright\arraybackslash}p{0.54\textwidth}@{}}
\caption{All %d enabled rules, generated from the rule metadata (\texttt{python3 -m tsw --list-rules}).
Type: hit, rate (Poisson), choice (binomial), stat (Gaussian), pred.\ (predicate).
Rules are named \texttt{family.rule}; the family prefix is given by the group headings. Dir: direction of a hit (for choice rules, of variant A). Cap: per-rule cap on $|w \cdot \mathrm{LLR}|$; w: weight.}
\label{tab-appendix}\\
\toprule
rule & type & dir & cap & w & description \\
\midrule
\endfirsthead
\toprule
rule & type & dir & cap & w & description \\
\midrule
\endhead
%s
\bottomrule
\end{longtable}}
""" % (len(eng.rules), "\n".join(rows)))


def build_data_table():
    d, dates = manifest_counts()
    rows = []
    for split in ("trn", "val", "test"):
        a, b, n = dates[split]
        cells = []
        for g in ("sent", "100", "500"):
            c = d["%s.%s.bal.jsonl" % (split, g)]
            cells.append("{:,}".format(c["o"]).replace(",", r"\,"))
        tok = d["%s.500.bal.jsonl" % split]
        rows.append(r"%s & %s -- %s & %s & %s & %s & %s & %s \\"
                    % (split, a, b, n, cells[0], cells[1], cells[2],
                       "{:,}".format(tok["tok_o"] + tok["tok_t"]).replace(",", r"\,")))
    write("tab_data.tex", r"""\begin{table}[tbp]
\centering\small
\caption{Date-based splits (balanced files; counts per class, so each set has twice as many units).
Sessions: first and last session date present and number of sessions. Tokens: whitespace tokens
of the balanced 500-token file, both classes. Source: \texttt{data/MANIFEST.md}.}
\label{tab-data}
\begin{tabular}{@{}l l r r r r r@{}}
\toprule
split & sessions & $n$ & sentences & 100-token & 500-token & tokens (500) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))


def build_results_table(res, hyb):
    cols = [("sent masked", "sent"), ("100 masked", "100"), ("500 masked", "500")]

    def line(label, split, cells):
        return r"%s & %s & %s \\" % (label, split, " & ".join(r"%s / %s" % c for c in cells))

    eng = "Default (integrated)"
    base = "baselines:MeanSentLen"
    rows = [line("rule engine", "trn", [res[(eng, "trn")][c] for c, _ in cols]),
            line("rule engine", "val", [res[(eng, "val")][c] for c, _ in cols]),
            line("mean sentence length", "trn", [res[(base, "trn")][c] for c, _ in cols]),
            line("mean sentence length", "val", [res[(base, "val")][c] for c, _ in cols])]
    # best hybrid per column on val (masked AUC)
    best = []
    for _, g in cols:
        m = max((k for k in hyb[g] if k not in ("MeanSentLen", "engine_raw", "bow_lr")),
                key=lambda k: float(strip_md(hyb[g][k]["val_masked"][1])))
        acc, auc = hyb[g][m]["val_masked"]
        best.append((f3(acc), f3(auc), m))
    rows.append(r"best expert-feature hybrid & val & %s \\"
                % " & ".join(r"%s / %s\textsuperscript{%s}" % (a, u, tex(m).replace(r"\_", r"\_"))
                             for a, u, m in best))
    write("tab_results.tex", r"""\begin{table}[tbp]
\centering\small
\caption{Accuracy / ROC-AUC on balanced, topic-masked data (unmasked figures differ by at most
.001). The engine and the baseline have no fitted parameters; trn is where the rule values were
read off, val is held out. The hybrid row is the best model per column among the ML models over
expert features (Section~\ref{sec-hybrid}; superscript: model). Sources: \texttt{docs/INTEGRATION.md},
\texttt{hybrid/reports/METRICS.md}.}
\label{tab-results}
\begin{tabular}{@{}l l c c c@{}}
\toprule
system & split & sentence & 100 tokens & 500 tokens \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))
    return best


def build_ablation_table():
    head, rows = integration_ablation()
    out = []
    for r in rows:
        cells = [tex(strip_md(c)) for c in r]
        if cells[0] == "(full Default)":
            cells[0] = "all families"
        out.append(" & ".join(cells) + r" \\")
        if r is rows[0]:
            out.append(r"\midrule")
    write("tab_ablation.tex", r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Leave-one-family-out ablation on trn (masked, balanced): accuracy / AUC without the family
and the change ($\Delta$acc / $\Delta$AUC) relative to the full engine. Source: \texttt{docs/INTEGRATION.md}.}
\label{tab-ablation}
\begin{tabular}{@{}l c c c c c c@{}}
\toprule
without & sentence & $\Delta$ & 100 tokens & $\Delta$ & 500 tokens & $\Delta$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(out))


HYB_ROWS = [
    ("engine_raw", "rule engine (no fitting)"),
    ("engine_calib", "engine score, LR threshold and scale"),
    ("stack_rules", "LR re-weighting of the 49 rule contributions"),
    ("stack_families", "LR over the 7 family subscores"),
    ("lr_engine", "LR over rule contributions + rule statistics"),
    ("gbt_engine", "GBT over rule contributions + rule statistics"),
    ("lr_expert", "LR over $\\sim$110 literature word-list features"),
    ("gbt", "GBT over the literature features"),
    ("hybrid_plus", "engine score + 10 literature features (LR)"),
    ("lr_all", "LR over rules + statistics + literature features"),
    ("gbt_all", "GBT over rules + statistics + literature features"),
    ("bow_lr", "LR over the 300 most frequent words (no expert input)"),
    ("MeanSentLen", "baseline: mean sentence length"),
]


def build_hybrid_table(hyb):
    rows = []
    for key, label in HYB_ROWS:
        cells = []
        for g in ("sent", "100", "500"):
            m = hyb[g][key]
            cells.append(f3(m["val_masked"][1]))
            if key == "bow_lr":
                cells[-1] += r" (%s)" % f3(m["val"][1])
        if key == "bow_lr":
            rows.append(r"\midrule")
        rows.append(r"%s & \texttt{%s} & %s \\" % (label, tex(key), " & ".join(cells)))
    write("tab_hybrid.tex", r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Val AUC (topic-masked) of models fitted on trn (hyper-parameters by session-grouped CV
inside trn) and scored once on val. For the bag-of-words reference, the unmasked AUC is in
parentheses. Source: \texttt{hybrid/reports/METRICS.md}.}
\label{tab-hybrid}
\begin{tabular}{@{}l l c c c@{}}
\toprule
model & key & sentence & 100 tokens & 500 tokens \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))


# Readable candidate rules from the shallow decision trees: (gran, leaf rank in the report).
TREE_LEAVES = [("500", 1), ("500", 4), ("100", 5)]


def tree_rules():
    """hybrid/reports/tree_rules.md: {gran: [(conditions, p_trans, n)]} for tree_d4."""
    md = read("hybrid/reports/tree_rules.md")
    out = {}
    for gran in ("sent", "100", "500"):
        sec = md[md.index("## tree_d4, gran = %s" % gran):].split("\n## ")[0]
        out[gran] = [(m.group(1), m.group(2), m.group(3)) for m in
                     re.finditer(r"^- IF (.*?) THEN P\(trans\) = ([\d.]+) \(n = (\d+)\)", sec, re.M)]
    return out


def build_tree_table():
    trees = tree_rules()
    rows = []
    for gran, k in TREE_LEAVES:
        cond, p, n = trees[gran][k - 1]
        parts = []
        for c in cond.split(" AND "):
            f, op, v = c.split(" ")
            parts.append(r"\texttt{%s} %s %s" % (tex(f), {"<=": r"$\le$", ">": "$>$"}[op], v))
        rows.append(r"%s & %s & %s & %s \\" % ("100 tokens" if gran == "100" else "500 tokens",
                                               r" $\wedge$ ".join(parts), p, "{:,}".format(int(n))))
    write("tab_trees.tex", r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Three leaves of the depth-4 decision trees over the literature features, fitted on trn,
read as candidate rules. \texttt{rate\_x}: hits per 1000 words (\texttt{p\_} = particle,
\texttt{poss\_nonrefl} = \emph{hans/hennes/deras}); \texttt{mean\_sent\_len}: words per sentence.
P(T): share of translated units in the leaf on trn; $n$: units in the leaf.
Source: \texttt{hybrid/reports/tree\_rules.md}.}
\label{tab-trees}
\begin{tabularx}{\textwidth}{@{}l X r r@{}}
\toprule
unit & condition & P(T) & $n$ \\
\midrule
%s
\bottomrule
\end{tabularx}
\end{table}
""" % "\n".join(rows))


def literature_check_counts():
    md = read("hybrid/reports/literature_check.md")
    head, rows = md_rows(md, "# Literature directions vs trn data")
    idx = head.index("verdict")
    c = {"confirmed": 0, "contradicted": 0, "weak/mixed": 0, "none": 0}
    for r in rows:
        v = strip_md(r[idx])
        if v.startswith("no prediction"):
            c["none"] += 1
        else:
            c[v] += 1
    return len(rows), c


def build_numbers(eng, stack):
    """LaTeX macros for numbers quoted in the prose that are derived, not copied."""
    lines = [r"\newcommand{\nRules}{%d}" % len(eng.rules),
             r"\newcommand{\nFamilies}{%d}" % len(eng.family_caps)]
    n_lex = len(next(r for r in eng.rules if r.name == "calques.lexicon").per_label)
    lines.append(r"\newcommand{\nCalques}{%d}" % n_lex)
    for g, tag in (("sent", "Sent"), ("100", "Hund"), ("500", "Five")):
        vs = list(stack[g].values())
        ms = [v["mult"] for v in vs]
        lines.append(r"\newcommand{\multMin%s}{%s}" % (tag, min(vs, key=lambda v: v["mult"])["mult_s"]))
        lines.append(r"\newcommand{\multMax%s}{%s}" % (tag, max(vs, key=lambda v: v["mult"])["mult_s"]))
        lines.append(r"\newcommand{\multNeg%s}{%d}" % (tag, sum(m < 0 for m in ms)))
        lines.append(r"\newcommand{\multIn%s}{%d}" % (tag, sum(0.5 <= m <= 2.0 for m in ms)))
    n, c = literature_check_counts()
    lines += [r"\newcommand{\litN}{%d}" % n, r"\newcommand{\litConfirmed}{%d}" % c["confirmed"],
              r"\newcommand{\litContradicted}{%d}" % c["contradicted"],
              r"\newcommand{\litWeak}{%d}" % c["weak/mixed"],
              r"\newcommand{\litNoPred}{%d}" % c["none"],
              r"\newcommand{\litPredicted}{%d}" % (n - c["none"])]
    write("numbers.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# figure

def build_figure(res, hyb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [0, 1, 2]
    xt = ["sentence", "100 tokens", "500 tokens"]
    cols = ["sent masked", "100 masked", "500 masked"]
    grans = ["sent", "100", "500"]
    eng = "Default (integrated)"
    base = "baselines:MeanSentLen"
    series = [
        ("rule engine", "#2a78d6", -5,
         [float(res[(eng, "trn")][c][1]) for c in cols],
         [float(res[(eng, "val")][c][1]) for c in cols]),
        ("hybrid LR, rules + literature features", "#eb6834", 6,
         [float(hyb[g]["lr_all"]["cv"]) for g in grans],
         [float(strip_md(hyb[g]["lr_all"]["val_masked"][1])) for g in grans]),
        ("mean sentence length", "#6f6e69", 0,
         [float(res[(base, "trn")][c][1]) for c in cols],
         [float(res[(base, "val")][c][1]) for c in cols]),
    ]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    # dy: label offset in points (engine and hybrid end within .006 at 500 tokens)
    for label, color, dy, trn, val in series:
        ax.plot(xs, trn, color=color, lw=1.4, ls=(0, (3, 2)), marker="o", ms=3.5, mfc="white")
        ax.plot(xs, val, color=color, lw=2.0, marker="o", ms=4)
        ax.annotate(label, (2, val[2]), xytext=(8, dy), textcoords="offset points",
                    va="center", color="#0b0b0b", fontsize=7.5, annotation_clip=False)
    ax.set_xticks(xs, xt)
    ax.set_xlim(-0.15, 2.15)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC (topic-masked)")
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="#52514e", lw=2.0, label="val"),
                       Line2D([], [], color="#52514e", lw=1.4, ls=(0, (3, 2)),
                              label="trn (hybrid: CV inside trn)")],
              loc="lower right", frameon=False, fontsize=7)
    # direct labels sit outside the axes on the right
    fig.subplots_adjust(left=0.12, right=0.58, bottom=0.12, top=0.97)
    os.makedirs(ASSETS, exist_ok=True)
    fig.savefig(os.path.join(ASSETS, "fig_auc.pdf"))
    fig.savefig(os.path.join(ASSETS, "fig_auc.png"), dpi=200)
    print("wrote assets/fig_auc.pdf, fig_auc.png")


# ---------------------------------------------------------------------------
# examples

def refresh_examples():
    want = {e["id"]: e["gran"] for e in EXAMPLES}
    out = {}
    for gran in sorted(set(want.values())):
        path = os.path.join(ROOT, "data", "processed", "trn.%s.bal.jsonl" % gran)  # trn only
        with open(path, encoding="utf-8") as f:
            for ln in f:
                rec = json.loads(ln)
                if rec["id"] in want and want[rec["id"]] == gran:
                    out[rec["id"]] = {"id": rec["id"], "split": "trn", "gran": gran,
                                      "label": rec["label"], "date": rec["date"],
                                      "text": rec["text"]}
    missing = set(want) - set(out)
    if missing:
        sys.exit("examples not found in trn: %s" % sorted(missing))
    with open(EX_FILE, "w", encoding="utf-8") as f:
        json.dump([out[e["id"]] for e in EXAMPLES], f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote %s" % os.path.relpath(EX_FILE, ROOT))


def highlight_text(text, spans, idx_of):
    """LaTeX for text with spans highlighted. Overlaps are resolved per character: the net
    signed share of all spans covering a character sets its colour (T/O) and intensity.
    A superscript after a span's last character gives the rule's number in the list."""
    cuts = {0, len(text)}
    for s in spans:
        cuts.update((s["start"], s["end"]))
    cuts = sorted(cuts)
    ends = {}
    for s in spans:
        ends.setdefault(s["end"], set()).add(idx_of[s["rule"]])
    out = []
    for a, b in zip(cuts, cuts[1:]):
        seg = text[a:b]
        net = sum(s["contribution_share"] for s in spans if s["start"] <= a and s["end"] >= b)
        cover = any(s["start"] <= a and s["end"] >= b for s in spans)
        if cover and abs(net) > 1e-9:
            pct = int(round(18 + 52 * min(abs(net), 2.0) / 2.0))
            col = "tsT" if net > 0 else "tsO"
            # highlight word by word so the box can break lines; spaces stay inside
            out.append(r"\highLight[%s!%d]{%s}" % (col, pct, tex(seg)))
        else:
            out.append(tex(seg))
        if b in ends:
            out.append(r"\textsuperscript{\sffamily\bfseries %s}" % ",".join(str(i) for i in sorted(ends[b])))
    return "".join(out)


def build_examples(eng):
    with open(EX_FILE, encoding="utf-8") as f:
        recs = {r["id"]: r for r in json.load(f)}
    rules = {r.name: r for r in eng.rules}
    boxes = []
    for k, ex in enumerate(EXAMPLES, 1):
        rec = recs[ex["id"]]
        assert rec["split"] == "trn" and rec["id"].startswith(("ep-", "c100:"))
        a = eng.annotate(rec["text"])
        truth, pred = rec["label"], int(a["score"] > 0)
        outcome = {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(truth, pred)]
        for rname, sign in ex["expect"].items():
            c = a["rules"].get(rname, 0.0)
            if not (c * sign > 0.05):
                sys.exit("example %s: note expects %s %s, engine gives %+.3f"
                         % (ex["id"], rname, "+" if sign > 0 else "-", c))
        # highlight token-level spans only: StatRule spans are whole sentences
        spans = [s for s in a["spans"] if not isinstance(rules[s["rule"]], StatRule)]
        contrib = sorted(a["rules"].items(), key=lambda kv: -abs(kv[1]))
        listed = [(n, c) for n, c in contrib if abs(c) >= 0.1][:7]
        span_rules = {s["rule"] for s in spans}
        listed_names = [n for n, _ in listed]
        for n in sorted(span_rules - set(listed_names), key=lambda n: -abs(a["rules"].get(n, 0))):
            listed.append((n, a["rules"].get(n, 0.0)))
        idx_of = {n: i for i, (n, _) in enumerate(listed, 1)}
        body = highlight_text(rec["text"], [s for s in spans if s["rule"] in idx_of], idx_of)
        rest = [c for n, c in a["rules"].items() if n not in idx_of]
        caps = [d for d in a["doc_level"] if d["rule"].startswith("cap:")]
        items = []
        for i, (n, c) in enumerate(listed, 1):
            hl = "" if n in span_rules else r" \textit{(no span)}"
            items.append(r"\textbf{%d}~\texttt{%s} & $%+.2f$ & %s%s \\"
                         % (i, tex(n), c, tex(rules[n].description), hl))
        tail = []
        if rest:
            tail.append(r"%d other rules: $%+.2f$ in total (mostly absence and length terms)"
                        % (len(rest), sum(rest)))
        for d in caps:
            tail.append(r"family cap \texttt{%s}: $%+.2f$" % (tex(d["family"]), d["contribution"]))
        gran = "sentence" if ex["gran"] == "sent" else "100-token chunk"
        title = (r"Example %d \quad %s \quad \textbf{%s} \quad true: %s \quad P(translated) = %.3f"
                 r" \quad score $%+.2f$ \quad %d words"
                 % (k, gran, outcome, "translated" if truth else "original", a["prob"],
                    a["score"], a["n_words"]))
        gloss = (r"\par\smallskip{\small\textit{`%s'}}" % tex(ex["gloss"])) if ex["gloss"] else ""
        boxes.append(r"""\begin{exbox}[%s]{%s}
{\setlength{\parindent}{0pt}\raggedright %s}%s
\par\medskip
{\footnotesize
\begin{tabularx}{\linewidth}{@{}l r X@{}}
%s
\end{tabularx}
%s}
\par\smallskip{\small\textbf{Note.} %s}
\end{exbox}
""" % ("unbreakable" if ex["gran"] == "sent" else "", title, body, gloss, "\n".join(items),
       (r"\par " + "; ".join(tail) + ".") if tail else "", ex["note"]))
        print("example %d %s %s score %+.3f" % (k, ex["id"], outcome, a["score"]))
    write("examples.tex", "\n".join(boxes))
    write("examples_ids.txt", "\n".join(e["id"] for e in EXAMPLES) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-examples", action="store_true",
                    help="re-extract the example texts from data/processed/trn.*.bal.jsonl")
    ap.add_argument("--no-figure", action="store_true", help="skip matplotlib")
    args = ap.parse_args()
    if args.refresh_examples:
        refresh_examples()
    eng = Engine()
    res = integration_results()
    hyb = hybrid_metrics()
    stack = stacking()
    build_numbers(eng, stack)
    build_data_table()
    build_feature_table(eng, stack)
    build_appendix(eng)
    build_results_table(res, hyb)
    build_ablation_table()
    build_hybrid_table(hyb)
    build_tree_table()
    build_examples(eng)
    if not args.no_figure:
        build_figure(res, hyb)


if __name__ == "__main__":
    main()

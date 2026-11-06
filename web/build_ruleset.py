"""Export the Python rule engine's data to web/site/ruleset.json for the JS port.

    python3 -I web/build_ruleset.py            # from the repository root

Everything data-like is read from the live Python objects (discover() and
the family modules), never retyped: rule metadata, parameters, derived
constants (log ratios computed by Python, so JS uses the very same floats),
label templates, word lists of the hand-written predicate rules, the text
normaliser's tables and every regex, translated by web/pyre2js.py.

Code that is not data (the rule-type maths, the normaliser, sentence splitter,
tokenizer, Doc views and the three PredicateRules) is ported by hand in
web/site/tsw.js. The builder refuses rules it does not know how to export
(an unknown StatRule unit function or PredicateRule), so a new rule of that
kind fails here instead of silently disappearing from the web checker.
"""
import json
import os
import subprocess
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pyre2js import translate  # noqa: E402
from tsw import text as T  # noqa: E402
from tsw.doc import Doc  # noqa: E402
from tsw.engine import Default  # noqa: E402
from tsw.rules import base, connpunct, stats, wordorder  # noqa: E402

OUT = os.path.join(HERE, "site", "ruleset.json")
PLACEHOLDER = ""   # stands for the matched surface in exported label templates


def rx(src, flags=0):
    js, groups = translate(src, flags)
    return {"py": src, "js": js}


def label_template(tmpl):
    """base._fmt_label(tmpl, surface) as a string with PLACEHOLDER for the surface."""
    if tmpl is None:
        return "'" + PLACEHOLDER + "'"
    try:
        return tmpl.format(m=PLACEHOLDER)
    except (KeyError, IndexError, ValueError):
        return tmpl


def matcher(m):
    d = {"rx": rx(m.source), "case": bool(m.case)}
    if m.labels:
        d["groups"] = {g: label_template(lab) for g, lab in m.labels.items()}
        d["groups_raw"] = dict(m.labels)   # ChoiceRule span labels use the raw key
    return d


STAT_UNITS = {
    base.sentence_lengths: "sentence_lengths",
    stats.long_word_units: "long_word_units",
    stats.av_nominal_units: "av_nominal_units",
    stats.av_other_units: "av_other_units",
}
PREDICATES = {"connpunct.exclamation", "wordorder.fronted_theme", "wordorder.subject_theme",
              "wordorder.v2_violation"}


def export_rule(r):
    d = {"name": r.name, "kind": r.kind, "family": r.family, "description": r.description,
         "direction": r.direction, "rationale": r.rationale, "source": r.source,
         "cap": r.cap, "weight": r.weight}
    if isinstance(r, base.HitRule):
        d["matcher"] = matcher(r.matcher)
        d["per_hit"] = r.per_hit
        d["label"] = label_template(r.label) if r.label is not None else None
        if r.per_label is not None:
            d["per_label"] = dict(r.per_label)
    elif isinstance(r, base.RateRule):
        d.update(matcher=matcher(r.matcher), rate_o=r.rate_o, rate_t=r.rate_t, log_ratio=r.log_ratio,
                 per_word=r.per_word, min_words=r.min_words,
                 label=label_template(r.label) if r.label is not None else None)
    elif isinstance(r, base.ChoiceRule):
        d.update(matcher_a=matcher(r.matcher_a), matcher_b=matcher(r.matcher_b), p_o=r.p_o, p_t=r.p_t,
                 la=r.la, lb=r.lb, label_a=r.label_a, label_b=r.label_b)
    elif isinstance(r, base.StatRule):
        if r.units not in STAT_UNITS:
            raise SystemExit("%s: unknown StatRule unit function %r: port it to tsw.js" % (r.name, r.units))
        d.update(units=STAT_UNITS[r.units], mu_o=r.mu_o, mu_t=r.mu_t, sd_o=r.sd_o, sd_t=r.sd_t,
                 unit_clip=list(r.unit_clip) if r.unit_clip else None, min_units=r.min_units,
                 unit_name=r.unit_name, fmt=r.fmt, span_min=r.span_min)
        if isinstance(r, stats.LengthStat):
            d.update(kind="lengthstat", tau=r.tau, scale=r.scale, unit_label=r.unit_label)
        elif type(r) is not base.StatRule:
            raise SystemExit("%s: unknown StatRule subclass %s" % (r.name, type(r).__name__))
    elif isinstance(r, base.PredicateRule):
        if r.name not in PREDICATES:
            raise SystemExit("%s: PredicateRule without a JS port in tsw.js" % r.name)
    else:
        raise SystemExit("%s: unknown rule type %s" % (r.name, type(r).__name__))
    return d


def ranges(pred):
    out = []
    for c in range(sys.maxunicode + 1):
        if pred(chr(c)):
            if out and out[-1][1] == c - 1:
                out[-1][1] = c
            else:
                out.append([c, c])
    return out


def text_data():
    table = {}
    for cp, rep in T._TABLE.items():
        table[chr(cp)] = rep
    return {
        "table": table,
        "no_split_abbr": sorted(T.NO_SPLIT_ABBR),
        "end_abbr": sorted(T.END_ABBR),
        "openers": T._OPENERS,
        "candidate": rx(T._CANDIDATE.pattern),
        "has_alnum": rx(T._HAS_ALNUM.pattern),
        "token": rx(T.TOKEN_RX.pattern),
        "address": rx(T.ADDRESS_RX.pattern),
        # unicodedata.combining(c) != 0 (used to cluster non-NFC input)
        "combining": ranges(lambda ch: unicodedata.combining(ch) != 0),
        # str.isdigit() code points that are not decimal digits (Nd): superscripts, circled digits ...
        "digit_extra": ranges(lambda ch: ch.isdigit() and not ch.isdecimal()),
        "unicode_version": unicodedata.unidata_version,
    }


def wordorder_data():
    W = wordorder
    sets = {k: sorted(getattr(W, k)) for k in (
        "SUBJ_PERS", "SUBJ_POST", "COPULA", "FIN", "AUX", "ADV1", "ADV_NO_V2_CHECK", "CONJ_ADV",
        "PREP", "SUBJN1", "V3_ADV", "_STOP", "_VOC_HEAD", "_INTERJ", "_COORD", "_NONVERB",
        "_FOCUS_ADV", "_NP_START", "_NOT_DEF")}
    seqs = {k: [list(s) for s in getattr(W, k)] for k in ("ADV_MULTI", "SUBJN_MULTI")}
    tuple_sets = {k: sorted(" ".join(s) for s in getattr(W, k))
                  for k in ("SUBJN_WITH_VERB", "SUBJN_INF", "_FOCUS_MULTI")}
    return {
        "sets": sets, "seqs": seqs, "tuple_sets": tuple_sets,
        "regex": {"_ADV_SUFFIX": rx(W._ADV_SUFFIX.pattern), "_FIN_SUFFIX": rx(W._FIN_SUFFIX.pattern),
                  "_DEF_SUFFIX": rx(W._DEF_SUFFIX.pattern),
                  "EMBEDDED_NEG_RX": rx(W.EMBEDDED_NEG_RX.pattern)},
        "OPENING_LLR": dict(W.OPENING_LLR), "OPENING_P": {k: list(v) for k, v in W.OPENING_P.items()},
        "_OPEN_ALIAS": dict(W._OPEN_ALIAS), "_OPEN_LABEL": dict(W._OPEN_LABEL),
        "SUBJECT_THEME": list(W.SUBJECT_THEME), "FRONTED_THEME": list(W.FRONTED_THEME),
        "V2_VIOLATION_HIT": dict(W.V2_VIOLATION_HIT), "_V2_LABEL": dict(W._V2_LABEL),
    }


def exclamation_data():
    # the span label lives inside a function: read it by running it on a probe text
    probe = Doc("Det är bra! Nu går vi vidare.")
    labels = {lab for _, _, _, lab in connpunct._excl_spans(probe)}
    if len(labels) != 1:
        raise SystemExit("connpunct.exclamation: could not read the span label")
    return {"end_rx": rx(connpunct._EXCL_END.pattern), "log": connpunct._EXCL_LOG,
            "per_word": connpunct._EXCL_PER_WORD, "label": labels.pop()}


def git_commit():
    try:
        # last commit that changed the engine (not HEAD: unrelated commits do not change the rules)
        return subprocess.check_output(["git", "-C", ROOT, "log", "-1", "--format=%h", "--", "tsw"],
                                       text=True).strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def main():
    eng = Default()
    rules = [export_rule(r) for r in eng.rules]
    families = []
    for r in eng.rules:
        if r.family not in families:
            families.append(r.family)
    out = {
        "meta": {"generated_by": "web/build_ruleset.py", "engine_commit": git_commit(),
                 "n_rules": len(rules), "placeholder": PLACEHOLDER,
                 "python_unicode_version": unicodedata.unidata_version},
        "prior": eng.prior,
        "families": [{"name": f, "cap": eng.family_caps[f]} for f in families],
        "text": text_data(),
        "predicates": {"wordorder": wordorder_data(), "exclamation": exclamation_data(),
                       "stats": {"PLACEHOLDER": stats.PLACEHOLDER, "LONG": stats.LONG,
                                 "_NOMINAL_SUFFIXES": list(stats._NOMINAL_SUFFIXES)}},
        "rules": rules,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print("wrote %s: %d rules in %d families, %d bytes"
          % (os.path.relpath(OUT, ROOT), len(rules), len(families), os.path.getsize(OUT)))


if __name__ == "__main__":
    main()

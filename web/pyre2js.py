"""Translate a Python `re` pattern into an equivalent JavaScript RegExp source.

The translation goes through Python's own parser (sre_parse), so every
construct is handled explicitly or rejected; nothing is rewritten textually.
The output is meant for the JavaScript `v` flag (Unicode sets mode), which
gives code-point matching, Unicode property escapes and nested classes.

Python (str patterns, Unicode mode) vs JavaScript, and what we emit:

  \\w  Python: str.isalnum() or '_'      ->  [\\p{L}\\p{N}_]
  \\d  Python: Unicode decimal (Nd)      ->  \\p{Nd}
  \\s  Python: str.isspace()             ->  explicit class of those code points
  \\b, \\B  Unicode-aware in Python; ASCII-only in JS even with u/v
                                        ->  lookaround over the \\w class
  .   any char except \\n                ->  [^\\n]
  ^   start of string (no MULTILINE)    ->  ^
  $   end, or before a final \\n         ->  (?=\\n?$)
  (?i:...) scoped case-insensitivity    ->  each literal expanded to the class
                                            of characters Python's IGNORECASE
                                            treats as equal to it
  (?P<name>...)                         ->  (?<name>...)
  lookbehind (fixed width in Python)    ->  same (JS allows any width)

Constructs that have no faithful translation (conditional groups, atomic
groups, possessive repeats, back-references, LOCALE/MULTILINE/DOTALL/VERBOSE
global flags under IGNORECASE ranges) raise TranslationError, so a future
rule that uses one fails loudly at build time.
"""
import re
import sys

try:  # Python 3.11+ moved these
    from re import _parser as sre_parse, _constants as C
except ImportError:  # pragma: no cover
    import sre_parse
    import sre_constants as C


class TranslationError(ValueError):
    pass


# code points Python's \s matches (str.isspace)
PY_SPACE = [c for c in range(sys.maxunicode + 1) if chr(c).isspace()]

WORD = r"[\p{L}\p{N}_]"
NOT_WORD = r"[^\p{L}\p{N}_]"
DIGIT = r"\p{Nd}"
NOT_DIGIT = r"\P{Nd}"

_SYNTAX = set("^$\\.*+?()[]{}|/")


def _ranges(cps):
    cps = sorted(set(cps))
    out = []
    for c in cps:
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return out


def _u(c):
    return "\\u{%X}" % c


def _class_body(cps):
    return "".join(_u(a) if a == b else "%s-%s" % (_u(a), _u(b)) for a, b in _ranges(cps))


SPACE = "[" + _class_body(PY_SPACE) + "]"
NOT_SPACE = "[^" + _class_body(PY_SPACE) + "]"

_CATEGORY = {
    C.CATEGORY_WORD: WORD, C.CATEGORY_NOT_WORD: NOT_WORD,
    C.CATEGORY_DIGIT: DIGIT, C.CATEGORY_NOT_DIGIT: NOT_DIGIT,
    C.CATEGORY_SPACE: SPACE, C.CATEGORY_NOT_SPACE: NOT_SPACE,
}

# --- case-insensitive equivalence classes, as Python's re computes them -------

_LOWER_INDEX = None


def _case_variants(c):
    """All code points x with re.fullmatch(re.escape(chr(c)), chr(x), re.I)."""
    global _LOWER_INDEX
    if _LOWER_INDEX is None:
        idx = {}
        for x in range(sys.maxunicode + 1):
            ch = chr(x)
            for key in {ch.lower(), ch.upper(), ch.casefold()}:
                idx.setdefault(key, set()).add(x)
        _LOWER_INDEX = idx
    ch = chr(c)
    cands = set()
    for key in {ch, ch.lower(), ch.upper(), ch.casefold()}:
        cands |= _LOWER_INDEX.get(key, set())
    cands.add(c)
    rx = re.compile(re.escape(ch), re.I)
    return sorted(x for x in cands if rx.fullmatch(chr(x)))


def _literal(c, icase):
    if icase:
        vs = _case_variants(c)
        if len(vs) > 1:
            return "[" + "".join(_class_char(v) for v in vs) + "]"
    ch = chr(c)
    if ch in _SYNTAX:
        return "\\" + ch
    if 0x20 <= c < 0x7F or (c > 0x7F and ch.isprintable() and not ch.isspace()):
        return ch
    return _u(c)


def _class_char(c):
    """One character inside a v-mode class: letters/digits literal, the rest escaped."""
    ch = chr(c)
    if ch.isalnum() and c < 0x10000:
        return ch
    return _u(c)


class _Emitter:
    def __init__(self, groupnames):
        self.names = {gid: name for name, gid in groupnames.items()}

    def seq(self, items, icase):
        return "".join(self.op(op, av, icase) for op, av in items)

    def atom(self, items, icase):
        """Emit a repeat operand as a single atom."""
        s = self.seq(items, icase)
        # every one of these emits a single atom: one char, an escape, a class or a group
        if len(items) == 1 and items[0][0] in (C.LITERAL, C.NOT_LITERAL, C.IN, C.ANY, C.SUBPATTERN, C.BRANCH):
            return s
        return "(?:%s)" % s

    def op(self, op, av, icase):
        if op == C.LITERAL:
            return _literal(av, icase)
        if op == C.NOT_LITERAL:
            vs = _case_variants(av) if icase else [av]
            return "[^" + "".join(_class_char(v) for v in vs) + "]"
        if op == C.ANY:
            return "[^\\n]"
        if op == C.IN:
            return self.cls(av, icase)
        if op == C.BRANCH:
            _, alts = av
            return "(?:%s)" % "|".join(self.seq(a, icase) for a in alts)
        if op == C.SUBPATTERN:
            gid, add, dele, p = av
            if add & ~(C.SRE_FLAG_IGNORECASE | C.SRE_FLAG_UNICODE) or dele & ~C.SRE_FLAG_IGNORECASE:
                raise TranslationError("unsupported scoped flags %r/%r" % (add, dele))
            inner_icase = (icase or bool(add & C.SRE_FLAG_IGNORECASE)) and not (dele & C.SRE_FLAG_IGNORECASE)
            body = self.seq(p, inner_icase)
            if gid is None:
                return "(?:%s)" % body
            name = self.names.get(gid)
            return "(?<%s>%s)" % (name, body) if name else "(%s)" % body
        if op in (C.MAX_REPEAT, C.MIN_REPEAT):
            lo, hi, item = av
            if hi == C.MAXREPEAT:
                q = "*" if lo == 0 else "+" if lo == 1 else "{%d,}" % lo
            elif lo == 0 and hi == 1:
                q = "?"
            elif lo == hi:
                q = "{%d}" % lo
            else:
                q = "{%d,%d}" % (lo, hi)
            if op == C.MIN_REPEAT:
                q += "?"
            return self.atom(item, icase) + q
        if op == C.AT:
            if av in (C.AT_BEGINNING, C.AT_BEGINNING_STRING):
                return "^"
            if av == C.AT_END:
                return "(?=\\n?$)"
            if av == C.AT_END_STRING:
                return "$"
            if av in (C.AT_BOUNDARY, C.AT_UNI_BOUNDARY):
                return "(?:(?<=%s)(?!%s)|(?<!%s)(?=%s))" % (WORD, WORD, WORD, WORD)
            if av in (C.AT_NON_BOUNDARY, C.AT_UNI_NON_BOUNDARY):
                return "(?:(?<=%s)(?=%s)|(?<!%s)(?!%s))" % (WORD, WORD, WORD, WORD)
            raise TranslationError("unsupported anchor %r" % av)
        if op in (C.ASSERT, C.ASSERT_NOT):
            direction, p = av
            kind = ("=" if op == C.ASSERT else "!")
            return "(?%s%s%s)" % ("<" if direction < 0 else "", kind, self.seq(p, icase))
        raise TranslationError("unsupported regex construct %r" % (op,))

    def cls(self, items, icase):
        neg = False
        parts = []
        for op, av in items:
            if op == C.NEGATE:
                neg = True
            elif op == C.LITERAL:
                vs = _case_variants(av) if icase else [av]
                parts.extend(_class_char(v) for v in vs)
            elif op == C.RANGE:
                if icase:
                    raise TranslationError("character range under IGNORECASE")
                a, b = av
                parts.append("%s-%s" % (_class_char(a), _class_char(b)))
            elif op == C.CATEGORY:
                if av not in _CATEGORY:
                    raise TranslationError("unsupported category %r" % av)
                s = _CATEGORY[av]
                parts.append(s)   # property escapes and nested classes are allowed in v mode
            else:
                raise TranslationError("unsupported class item %r" % (op,))
        return "[%s%s]" % ("^" if neg else "", "".join(parts))


def translate(src, flags=0):
    """Python regex source -> (JS source for the v flag, {group name: index})."""
    p = sre_parse.parse(src, flags)
    gflags = p.state.flags
    bad = gflags & ~(C.SRE_FLAG_UNICODE | C.SRE_FLAG_IGNORECASE)
    if bad:
        raise TranslationError("unsupported global flags %r" % bad)
    lo, _ = p.getwidth()
    if lo == 0:
        raise TranslationError("pattern can match the empty string (finditer semantics differ): %r" % src)
    em = _Emitter(p.state.groupdict)
    return em.seq(p.data, bool(gflags & C.SRE_FLAG_IGNORECASE)), dict(p.state.groupdict)

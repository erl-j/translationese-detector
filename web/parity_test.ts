// Parity test: the JS engine in web/site/ (the exact files the site serves)
// against the Python engine's results exported by web/export_parity.py.
//
//   deno test --allow-read web/parity_test.ts
//   PARITY_VERBOSE=1 deno test --allow-read --allow-env web/parity_test.ts   (more examples)
//
// Per text: score and P(translated) (relative tolerance 1e-6), per-rule
// contributions, per-family raw/clipped sums, every highlighted span (offsets,
// rule, direction, label, share), document-level evidence, the evidence
// sentences, word/sentence counts, the normalised text and sentence spans,
// and, for every rule regex, its match spans on the Python-normalised text
// (regex translation tested on its own). Plus a sweep over all code points of
// the character predicates the port uses. Python offsets are code points; they
// are converted to UTF-16 units before comparing.

import { createEngine } from "./site/tsw.js";

const HERE = new URL(".", import.meta.url);
const RS = JSON.parse(await Deno.readTextFile(new URL("site/ruleset.json", HERE)));
const DATA = JSON.parse(await Deno.readTextFile(new URL("parity/cases.json", HERE)));
let VERBOSE = false;
try {
  VERBOSE = !!Deno.env.get("PARITY_VERBOSE");
} catch (_) { /* no --allow-env */ }

const eng = createEngine(RS);
const I = eng._internal;
const TOL = 1e-6;

function close(a: number, b: number) {
  if (a === b) return true;
  return Math.abs(a - b) <= TOL * Math.max(1, Math.abs(a), Math.abs(b));
}

// code-point offset -> UTF-16 offset in string s
function cpMap(s: string): number[] {
  const m = [0];
  let u = 0;
  for (const ch of s) {
    u += ch.length;
    m.push(u);
  }
  return m;
}

// JS source of each regex, keyed like export_parity.py (rule:attr or family:name)
function regexIndex(): Map<string, string> {
  const out = new Map<string, string>();
  for (const r of RS.rules) {
    for (const attr of ["matcher", "matcher_a", "matcher_b"]) {
      if (r[attr]) out.set(`${r.name}:${attr}`, r[attr].rx.js);
    }
  }
  out.set("wordorder:EMBEDDED_NEG_RX", RS.predicates.wordorder.regex.EMBEDDED_NEG_RX.js);
  out.set("text:token", RS.text.token.js);
  out.set("text:candidate", RS.text.candidate.js);
  out.set("exclamation:end_rx", RS.predicates.exclamation.end_rx.js);
  return out;
}

type Fail = { id: string; what: string; detail: string };

Deno.test("JS engine matches Python on the exported cases", () => {
  const cases = DATA.cases;
  const rxIdx = regexIndex();
  const keys: string[] = DATA.meta.regex_keys;
  const missing = keys.filter((k) => !rxIdx.has(k));
  if (missing.length) throw new Error("regexes missing from ruleset: " + missing.join(" | "));
  const compiled = new Map(keys.map((k) => [k, new RegExp(rxIdx.get(k)!, "gv")]));
  const caseOf = new Map<string, boolean>();
  for (const r of RS.rules) {
    for (const attr of ["matcher", "matcher_a", "matcher_b"]) if (r[attr]) caseOf.set(`${r.name}:${attr}`, r[attr].case);
  }
  for (const k of ["text:token", "text:candidate", "exclamation:end_rx"]) caseOf.set(k, true);
  caseOf.set("wordorder:EMBEDDED_NEG_RX", false);

  const fails: Fail[] = [];
  const byWhat = new Map<string, number>();
  let exactScore = 0, nScored = 0, failedCases = 0, nSpans = 0, nRegexHits = 0;
  const bySrc = new Map<string, [number, number]>();
  let maxRel = 0;
  const t0 = performance.now();

  for (const c of cases) {
    const local: Fail[] = [];
    const fail = (what: string, detail: string) => local.push({ id: c.id, what, detail });
    let a: any;
    try {
      a = eng.annotate(c.text);
    } catch (e) {
      if (!c.error) fail("exception", String(e));
    }
    if (c.error) {
      if (a !== undefined) fail("exception", "Python raised " + c.error + ", JS did not");
    } else if (a !== undefined) {
      const py = c.py;
      const rawMap = cpMap(c.text);
      const normMap = cpMap(py.norm);
      const doc = new eng.Doc(c.text);
      if (doc.text !== py.norm) fail("normalise", JSON.stringify([doc.text.slice(0, 80), py.norm.slice(0, 80)]));
      if (doc.lower !== py.lower) fail("lower", "");
      const ss = py.sent_spans.map(([s, e]: number[]) => [normMap[s], normMap[e]]);
      if (JSON.stringify(ss) !== JSON.stringify(doc.sent_spans)) fail("sentences", JSON.stringify([ss.length, doc.sent_spans.length]));
      if (a.n_words !== py.n_words) fail("n_words", `${a.n_words} vs ${py.n_words}`);
      if (a.n_sentences !== py.n_sentences) fail("n_sentences", `${a.n_sentences} vs ${py.n_sentences}`);
      if (JSON.stringify(doc.is_address) !== JSON.stringify(py.is_address)) fail("is_address", "");
      // regexes on Python's own normalised strings
      const lowMap = cpMap(py.lower);
      for (const k of keys) {
        const re = compiled.get(k)!;
        const target = caseOf.get(k) ? py.norm : py.lower;
        const map = caseOf.get(k) ? normMap : lowMap;
        const want = (py.regex[k] || []).map(([s, e]: number[]) => [map[s], map[e]]);
        const got = [...I.finditer(re, target)].map((m: RegExpExecArray) => [m.index, m.index + m[0].length]);
        nRegexHits += want.length;
        if (JSON.stringify(want) !== JSON.stringify(got)) {
          fail("regex", k.slice(0, 70) + " " + JSON.stringify([want.slice(0, 3), got.slice(0, 3)]));
        }
      }
      // score
      nScored++;
      if (a.score === py.score) exactScore++;
      maxRel = Math.max(maxRel, Math.abs(a.score - py.score) / Math.max(1, Math.abs(py.score)));
      if (!close(a.score, py.score)) fail("score", `${a.score} vs ${py.score}`);
      if (!close(a.prob, py.prob)) fail("prob", `${a.prob} vs ${py.prob}`);
      if (a.label !== py.label) fail("label", `${a.label} vs ${py.label}`);
      const rk = Object.keys(a.rules), pk = Object.keys(py.rules);
      if (JSON.stringify(rk) !== JSON.stringify(pk)) fail("rules fired", JSON.stringify([rk, pk]));
      for (const k of pk) {
        if (!(k in a.rules) || !close(a.rules[k], py.rules[k])) fail("rule contribution", `${k}: ${a.rules[k]} vs ${py.rules[k]}`);
      }
      for (const f in py.families) {
        const x = a.families[f], y = py.families[f];
        if (!x || !close(x.raw, y.raw) || !close(x.clipped, y.clipped)) fail("family", f);
      }
      // spans
      nSpans += py.spans.length;
      if (a.spans.length !== py.spans.length) {
        fail("span count", `${a.spans.length} vs ${py.spans.length}`);
      } else {
        py.spans.forEach((s: any, i: number) => {
          const j = a.spans[i];
          const st = rawMap[s.start], en = rawMap[s.end];
          if (j.start !== st || j.end !== en || j.rule !== s.rule || j.direction !== s.direction) {
            fail("span", JSON.stringify([[j.start, j.end, j.rule], [st, en, s.rule]]));
          } else if (j.label !== s.label) {
            fail("span label", JSON.stringify([j.label, s.label]));
          } else if (!close(j.contribution_share, s.contribution_share)) {
            fail("span share", `${s.rule} ${j.contribution_share} vs ${s.contribution_share}`);
          } else if (j.text !== c.text.slice(st, en)) {
            fail("span text", "");
          }
        });
      }
      // document level
      const dl = (x: any[]) => x.map((d) => d.rule).join(",");
      if (dl(a.doc_level) !== dl(py.doc_level)) {
        fail("doc_level rules", JSON.stringify([dl(a.doc_level), dl(py.doc_level)]));
      } else {
        py.doc_level.forEach((d: any, i: number) => {
          const j = a.doc_level[i];
          if (!close(j.contribution, d.contribution)) fail("doc_level value", d.rule);
          if (j.label !== d.label) fail("doc_level label", JSON.stringify([j.label, d.label]));
        });
      }
      for (const k of ["evidence_translated", "evidence_original"]) {
        if (JSON.stringify(a[k]) !== JSON.stringify(py[k])) fail("evidence text", JSON.stringify([a[k], py[k]]));
      }
    }
    const src = c.src;
    const v = bySrc.get(src) || [0, 0];
    v[0]++;
    if (local.length) {
      failedCases++;
      v[1]++;
      for (const f of local) byWhat.set(f.what, (byWhat.get(f.what) || 0) + 1);
      fails.push(...local);
    }
    bySrc.set(src, v);
  }

  const secs = ((performance.now() - t0) / 1000).toFixed(1);
  console.log(`\nparity: ${cases.length - failedCases}/${cases.length} cases match (${secs}s)`);
  for (const [src, [n, f]] of bySrc) console.log(`  ${src.padEnd(18)} ${n - f}/${n}`);
  console.log(`  score bit-identical: ${exactScore}/${nScored}; max relative score difference ${maxRel.toExponential(2)}`);
  console.log(`  spans compared: ${nSpans}; regex matches compared: ${nRegexHits} over ${keys.length} regexes`);
  if (fails.length) {
    console.log("  mismatches by kind:", Object.fromEntries(byWhat));
    for (const f of fails.slice(0, VERBOSE ? 200 : 15)) console.log(`  - ${f.id} [${f.what}] ${f.detail.slice(0, 300)}`);
    throw new Error(`${failedCases} of ${cases.length} cases differ`);
  }
});

Deno.test("character predicates agree with Python on every code point", () => {
  const U = DATA.meta.unicode_tables;
  const inR = (r: number[][], cp: number) => {
    let lo = 0, hi = r.length - 1;
    while (lo <= hi) {
      const m = (lo + hi) >> 1;
      if (cp < r[m][0]) hi = m - 1;
      else if (cp > r[m][1]) lo = m + 1;
      else return true;
    }
    return false;
  };
  const W = /^[\p{L}\p{N}_]$/v;
  // the translated \s class, as it appears in the vocative regex
  const spaceClass = RS.text.address.js.match(/\[\\u\{9\}[^\]]*\]/)[0];
  const SP = new RegExp("^" + spaceClass + "$", "v");
  const checks: [string, (ch: string, cp: number) => boolean][] = [
    ["isalnum", (ch) => I.isAlnumStart(ch)],
    ["isalpha", (ch) => I.isAlphaCh(ch)],
    ["isupper", (ch) => I.isUpperCh(ch)],
    ["isdigit", (_ch, cp) => I.isDigitCp(cp)],
    ["word", (ch) => W.test(ch)],
    ["space", (ch) => SP.test(ch)],
  ];
  let total = 0, assigned = 0;
  const report: string[] = [];
  for (const [name, f] of checks) {
    const r = U[name];
    const diffs: number[] = [];
    for (let cp = 0; cp <= 0x10ffff; cp++) {
      if (cp >= 0xd800 && cp <= 0xdfff) continue; // lone surrogates never occur in matched text
      const ch = String.fromCodePoint(cp);
      if (f(ch, cp) !== inR(r, cp)) diffs.push(cp);
    }
    const asg = diffs.filter((cp) => !inR(U.unassigned, cp));
    total += diffs.length;
    assigned += asg.length;
    report.push(`  ${name.padEnd(8)} ${diffs.length} differing code points, ${asg.length} of them assigned in Python's Unicode ${DATA.meta.unicode}` +
      (asg.length ? " e.g. " + asg.slice(0, 8).map((c) => "U+" + c.toString(16).toUpperCase()).join(" ") : ""));
  }
  // lower(): per code point
  const lowDiff: number[] = [];
  for (let cp = 0; cp <= 0x10ffff; cp++) {
    if (cp >= 0xd800 && cp <= 0xdfff) continue;
    const ch = String.fromCodePoint(cp);
    const want = U.lower[String(cp)] ?? ch;
    if (ch.toLowerCase() !== want && !inR(U.unassigned, cp)) lowDiff.push(cp);
  }
  report.push(`  lower()   ${lowDiff.length} assigned code points differ` +
    (lowDiff.length ? " e.g. " + lowDiff.slice(0, 8).map((c) => "U+" + c.toString(16).toUpperCase()).join(" ") : ""));
  console.log("\ncode-point sweep (differences on unassigned-in-Python code points are Unicode-version drift):\n" + report.join("\n"));
  if (assigned + lowDiff.length > 0) throw new Error(`${assigned + lowDiff.length} assigned code points differ`);
});

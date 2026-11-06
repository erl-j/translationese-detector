// app.js: page logic for the Swedish Translationese Checker (no dependencies).
import { createEngine } from "./tsw.js";

const EXAMPLES = {
  // hand-written for this page (not corpus text)
  translated:
    "Herr talman! Först av allt skulle jag vilja tacka föredraganden för hennes hårda arbete. Det faktum att " +
    "kommissionen kommer att lägga fram ett nytt förslag är i synnerhet välkommet. Slutligen, vi måste säkerställa " +
    "att våra medborgare får inflytande på de beslut som påverkar deras vardag. I det förflutna har vi inte alltid " +
    "lyckats med detta, men jag ser fram emot att arbeta med er på daglig basis.",
  original:
    "Herr talman! Jag tycker att det här är en viktig fråga. Vi måste ju också se till att reglerna följs, och det " +
    "har vi inte alltid gjort. Därför anser jag att kommissionens förslag är bra, men det finns nämligen ett " +
    "problem: man har inte tagit hänsyn till de små företagen. Det vet vi alla. Dessutom är det ändå så att " +
    "medlemsstaterna bl.a. måste ta ett större ansvar, dvs. de måste genomföra det som vi redan har beslutat.",
};

const FAMILY_NAMES = {
  starter: "sentence length & modal particles",
  connpunct: "connectives & punctuation",
  particles: "particles & pronouns",
  register: "register & word choice",
  calques: "English calques",
  wordorder: "word order",
  stats: "text statistics",
};
const UNCLEAR = 0.5; // |score| below this: "Unclear"

const $ = (id) => document.getElementById(id);
const ta = $("text");
const statusEl = $("status");
let engine = null;
let ruleInfo = new Map();
let lastSpans = [];

const sign = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2);
const toward = (x) => (x > 0 ? "toward translated" : "toward original");
const strength = (x) => {
  const a = Math.abs(x);
  return a < 0.3 ? 1 : a < 0.7 ? 2 : a < 1.2 ? 3 : 4;
};
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}

async function init() {
  try {
    const res = await fetch("ruleset.json");
    if (!res.ok) throw new Error("ruleset.json: HTTP " + res.status);
    const RS = await res.json();
    engine = createEngine(RS);
    for (const r of RS.rules) ruleInfo.set(r.name, r);
    $("commit").textContent = `${RS.meta.engine_commit} (${RS.meta.n_rules} rules)`;
    statusEl.textContent = "Analysis updates as you type.";
  } catch (e) {
    const old = e instanceof SyntaxError && /flag|regular expression/i.test(String(e));
    statusEl.textContent = old
      ? "Your browser is too old for this page (it needs RegExp v-flag support: Chrome/Edge 112+, Firefox 116+, Safari 17+)."
      : "Could not load the rules: " + e.message;
    console.error(e);
    return;
  }
  if (!ta.value.trim()) ta.value = EXAMPLES.translated;
  analyse();
}

function analyse() {
  if (!engine) return;
  const text = ta.value;
  let a;
  try {
    a = engine.annotate(text, Infinity);
  } catch (e) {
    statusEl.textContent = "Analysis failed: " + e.message;
    console.error(e);
    return;
  }
  renderVerdict(a);
  renderHighlights(text, a.spans);
  renderEvidence(a);
}

function renderVerdict(a) {
  const lab = $("v-label");
  lab.className = "verdict-label";
  if (a.n_words === 0) {
    lab.textContent = "Type or paste Swedish text";
    lab.classList.add("is-u");
  } else if (Math.abs(a.score) < UNCLEAR) {
    lab.textContent = "Unclear";
    lab.classList.add("is-u");
  } else if (a.score > 0) {
    lab.textContent = "Translated from English";
    lab.classList.add("is-t");
  } else {
    lab.textContent = "Original Swedish";
    lab.classList.add("is-o");
  }
  const p = a.n_words ? a.prob : 0.5;
  $("meter-mark").style.left = (p * 100).toFixed(1) + "%";
  const m = $("meter");
  m.setAttribute("aria-valuenow", p.toFixed(3));
  m.setAttribute("aria-valuetext", `P(translated) ${(p * 100).toFixed(0)} percent`);
  $("v-prob").textContent = a.n_words ? p.toFixed(2) : "–";
  $("v-score").textContent = a.n_words ? sign(a.score) : "–";
  $("v-words").textContent = String(a.n_words);
  $("v-sents").textContent = String(a.n_sentences);
  $("v-note").classList.toggle("emph", a.n_words > 0 && a.n_words < 100);
}

// --- highlighted text: per character, the strongest covering span wins --------
function renderHighlights(text, spans) {
  lastSpans = spans;
  const box = $("highlighted");
  box.replaceChildren();
  hideTip(true);
  if (!text) return;
  const best = new Int32Array(text.length).fill(-1);
  spans.forEach((s, i) => {
    const w = Math.abs(s.contribution_share);
    for (let k = s.start; k < s.end; k++) {
      const b = best[k];
      if (b < 0 || w > Math.abs(spans[b].contribution_share)) best[k] = i;
    }
  });
  const frag = document.createDocumentFragment();
  let k = 0;
  while (k < text.length) {
    const b = best[k];
    let j = k + 1;
    while (j < text.length && best[j] === b) j++;
    const chunk = text.slice(k, j);
    if (b < 0) {
      frag.append(chunk);
    } else {
      const s = spans[b];
      const d = s.direction === "T" ? "t" : "o";
      const info = ruleInfo.get(s.rule);
      const mk = el("mark", {
        class: `${d}${strength(s.contribution_share)} dir-${d}`,
        tabindex: "0",
        "data-pos": String(k),
        "aria-label": `${chunk}: ${s.label}, ${sign(s.contribution_share)} ${toward(s.contribution_share)}` +
          (info ? ` (${info.description})` : ""),
      }, chunk);
      frag.append(mk);
    }
    k = j;
  }
  box.append(frag);
}

const tip = $("tip");
let pinned = null;

function showTip(mark) {
  const pos = +mark.dataset.pos;
  const cover = lastSpans.filter((s) => s.start <= pos && pos < s.end)
    .sort((x, y) => Math.abs(y.contribution_share) - Math.abs(x.contribution_share));
  const ul = el("ul");
  for (const s of cover) {
    const info = ruleInfo.get(s.rule);
    const d = s.direction === "T" ? "t" : "o";
    ul.append(el("li", {},
      el("div", { text: s.label }),
      el("div", {}, el("span", { class: `c ${d}`, text: sign(s.contribution_share) }), " " + toward(s.contribution_share)),
      el("div", { class: "r", text: (info ? info.description : s.rule) + " · " + (FAMILY_NAMES[s.family] || s.family) })));
  }
  tip.replaceChildren(ul);
  tip.hidden = false;
  const r = mark.getBoundingClientRect();
  const host = tip.offsetParent ? tip.offsetParent.getBoundingClientRect() : { left: 0, top: 0 };
  const tw = tip.offsetWidth;
  let left = r.left - host.left;
  const maxLeft = document.documentElement.clientWidth - tw - 16 - host.left;
  left = Math.max(16 - host.left, Math.min(left, maxLeft));
  tip.style.left = left + "px";
  tip.style.top = (r.bottom - host.top + 6) + "px";
}
function hideTip(force) {
  if (pinned && !force) return;
  pinned = null;
  tip.hidden = true;
}

const box = $("highlighted");
box.addEventListener("mouseover", (e) => {
  const m = e.target.closest("mark");
  if (m && !pinned) showTip(m);
});
box.addEventListener("mouseout", (e) => {
  const m = e.target.closest("mark");
  if (m && !pinned && !m.contains(e.relatedTarget)) hideTip();
});
box.addEventListener("focusin", (e) => {
  const m = e.target.closest("mark");
  if (m) { pinned = null; showTip(m); }
});
box.addEventListener("focusout", () => hideTip(true));
box.addEventListener("click", (e) => {
  const m = e.target.closest("mark");
  if (!m) return;
  if (pinned === m) { hideTip(true); return; }
  pinned = m;
  showTip(m);
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTip(true); });
document.addEventListener("click", (e) => { if (pinned && !e.target.closest("mark")) hideTip(true); });

// --- evidence lists -------------------------------------------------------------
const MINOR = 0.05; // items below this (and without highlighted words) are folded away

function minorBlock(n, total, mkList) {
  const det = el("details", {}, el("summary", { text: `${n} minor item${n === 1 ? "" : "s"} (together ${sign(total)})` }));
  det.append(mkList());
  return el("li", { class: "minor" }, det);
}

function renderEvidence(a) {
  const byRule = new Map();
  for (const s of a.spans) {
    if (!byRule.has(s.rule)) byRule.set(s.rule, new Map());
    const m = byRule.get(s.rule);
    m.set(s.label, (m.get(s.label) || 0) + 1);
  }
  const item = (e) => {
    const d = e.contribution > 0 ? "t" : "o";
    const info = ruleInfo.get(e.rule) || {};
    const clues = byRule.get(e.rule);
    let cluesEl = null;
    if (clues && clues.size) {
      const parts = [...clues].sort((x, y) => y[1] - x[1]);
      const shown = parts.slice(0, 6).map(([l, n]) => l + (n > 1 ? ` ×${n}` : ""));
      if (parts.length > 6) shown.push(`and ${parts.length - 6} more`);
      cluesEl = el("p", { class: "clues", text: shown.join("; ") });
    }
    return el("li", {},
      el("div", { class: "row" },
        el("span", { class: "desc", text: e.description }),
        el("span", { class: `val ${d}`, text: sign(e.contribution) })),
      cluesEl,
      el("details", {},
        el("summary", { text: "details" }),
        el("p", { class: "why", text: e.describe }),
        info.rationale ? el("p", { class: "why", text: "Why: " + info.rationale }) : null,
        el("p", { class: "fam", text: `Rule ${e.rule} · family: ${FAMILY_NAMES[e.family] || e.family}` })));
  };
  const fill = (listEl, items, sumEl) => {
    listEl.replaceChildren();
    let total = 0;
    const main = [], minor = [];
    for (const e of items) {
      total += e.contribution;
      (Math.abs(e.contribution) >= MINOR || byRule.has(e.rule) ? main : minor).push(e);
    }
    for (const e of main) listEl.append(item(e));
    if (minor.length) {
      const t = minor.reduce((s, e) => s + e.contribution, 0);
      listEl.append(minorBlock(minor.length, t, () => el("ol", { class: "ev-list" }, ...minor.map(item))));
    }
    if (!items.length) listEl.append(el("li", { class: "empty", text: "No rule fired on this side." }));
    sumEl.textContent = items.length ? `(${sign(total)} before family caps)` : "";
  };
  const ev = a.evidence; // sorted by |contribution|
  fill($("ev-t"), ev.filter((e) => e.contribution > 0), $("sum-t"));
  fill($("ev-o"), ev.filter((e) => e.contribution < 0), $("sum-o"));

  const docEl = $("ev-doc");
  docEl.replaceChildren();
  const docItem = (d) => {
    const name = d.rule.startsWith("cap:")
      ? `Family cap (${FAMILY_NAMES[d.family] || d.family}): ${d.label}`
      : d.label.replace(/ -> [+-][\d.]+ toward \w+$/, "");
    return el("li", {},
      el("span", { text: name }),
      el("span", { class: `val ${d.contribution > 0 ? "t" : "o"}`, text: sign(d.contribution) }));
  };
  const dl = [...a.doc_level].sort((x, y) => Math.abs(y.contribution) - Math.abs(x.contribution));
  const big = dl.filter((d) => Math.abs(d.contribution) >= MINOR || d.rule.startsWith("cap:"));
  const small = dl.filter((d) => !big.includes(d));
  for (const d of big) docEl.append(docItem(d));
  if (small.length) {
    const t = small.reduce((s, d) => s + d.contribution, 0);
    docEl.append(minorBlock(small.length, t, () => el("ul", { class: "doc-list" }, ...small.map(docItem))));
  }
  if (!dl.length) docEl.append(el("li", { class: "empty", text: "None." }));
}

// --- wiring ----------------------------------------------------------------------
let timer = null;
ta.addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(analyse, ta.value.length > 20000 ? 800 : 250);
});
$("analyse").addEventListener("click", () => { clearTimeout(timer); analyse(); });
$("ex-t").addEventListener("click", () => { ta.value = EXAMPLES.translated; analyse(); });
$("ex-o").addEventListener("click", () => { ta.value = EXAMPLES.original; analyse(); });
$("clear").addEventListener("click", () => { ta.value = ""; analyse(); ta.focus(); });
ta.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { clearTimeout(timer); analyse(); }
});

init();

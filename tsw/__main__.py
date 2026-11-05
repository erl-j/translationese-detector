"""Command line: classify one text and explain the verdict.

    python3 -m tsw "Herr talman! Det är ju så att ..."
    python3 -m tsw -f speech.txt
    echo "..." | python3 -m tsw -f -
    python3 -m tsw --json -f speech.txt            # Engine.annotate() as JSON
    python3 -m tsw --families starter --exclude starter.sent_len "..."
    python3 -m tsw --list-rules
"""
import argparse
import sys

from .engine import Engine, to_json


def highlight(text, spans):
    """Inline markup: [[surface]]{T|O rule +0.12}. Overlapping spans: first wins."""
    out, pos = [], 0
    for s in spans:
        if s["start"] < pos:
            continue
        out.append(text[pos:s["start"]])
        out.append("[[%s]]{%s %s %+.2f}" % (text[s["start"]:s["end"]], s["direction"],
                                           s["rule"], s["contribution_share"]))
        pos = s["end"]
    out.append(text[pos:])
    return "".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m tsw", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="text to classify")
    ap.add_argument("-f", "--file", help="read the text from a file ('-' = stdin)")
    ap.add_argument("--json", action="store_true", help="print Engine.annotate() as JSON")
    ap.add_argument("--families", help="only these families (comma-separated)")
    ap.add_argument("--exclude", help="drop these rules/families (comma-separated)")
    ap.add_argument("--all-rules", action="store_true", help="include rules disabled by default")
    ap.add_argument("--no-spans", action="store_true", help="do not print the highlighted text")
    ap.add_argument("--list-rules", action="store_true", help="list the active rules and exit")
    args = ap.parse_args(argv)

    eng = Engine(families=args.families, exclude=args.exclude, include_disabled=args.all_rules)
    if args.list_rules:
        for m in eng.describe_rules():
            print("%-34s %-9s %-10s dir=%s cap=%.1f w=%.2f  %s" % (
                m["name"], m["kind"], m["family"], m["direction"], m["cap"], m["weight"], m["description"]))
        return
    if args.file:
        text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    elif args.text is not None:
        text = args.text
    else:
        ap.error("give a text or -f FILE")

    a = eng.annotate(text)
    if args.json:
        print(to_json(a))
        return
    print("VERDICT: %s  (score %+.2f, P(translated) = %.2f; %d words, %d sentences)"
          % (a["label"].upper(), a["score"], a["prob"], a["n_words"], a["n_sentences"]))
    for side, key in (("translated", "evidence_translated"), ("original", "evidence_original")):
        if a[key]:
            print("\nEvidence for %s:" % side)
            for line in a[key]:
                print("  - " + line)
    caps = [d for d in a["doc_level"] if d["rule"].startswith("cap:")]
    for d in caps:
        print("  ! " + d["label"])
    if not a["rules"]:
        print("\nNo rule fired.")
    if a["spans"] and not args.no_spans:
        print("\nHighlighted:\n" + highlight(text, a["spans"]))


if __name__ == "__main__":
    main()

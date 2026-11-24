"""A probe: how long the library's lines are, against what the model reads of a line, and how long the lines are that
judges and users graded (results-sefaria.md, Patot: a chunker groups short segments and splits long ones).

    python eval-pool/probes/line_lengths.py --db seforim.db [--grades name=grades.tsv ...] [--skip-library]

Run from the repository's root. A line's words are counted as the model's text of it (pardes_semantic/clean.py). WINDOW
words is about the model's 512 tokens of a line (1.23 tokens a word, results-google.md); TRAINED is what the training
read of a passage (build_train_pairs.py's MAX_PASSAGE_WORDS). A grades.tsv has lineId and grade; judge/grades.tsv's
`from` column tells the sources the judges knew from the lines of the page.
"""
import argparse
import collections
import csv
import sqlite3
import sys
from pathlib import Path

sys.path[:0] = ["."]
from pardes_semantic.clean import clean  # noqa: E402

SHORT, TRAINED, WINDOW = 8, 220, 416
EDGES = [2, SHORT, 20, 50, 100, TRAINED, WINDOW, 1000, 10 ** 9]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--grades", action="append", default=[], help="<name>=<grades.tsv>; repeatable")
    ap.add_argument("--skip-library", action="store_true")
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    if not args.skip_library:
        lines, words = collections.Counter(), collections.Counter()
        n = total = cut = 0
        for (content,) in con.execute("SELECT content FROM line"):
            w = len(clean(content or "").split())
            n += 1
            total += w
            edge = next(e for e in EDGES if w <= e)
            lines[edge] += 1
            words[edge] += w
            cut += max(0, w - WINDOW)
        print(f"{n} lines, {total} words")
        low = 0
        for e in EDGES:
            print(f"{low + 1}-{e if e < 10 ** 9 else 'more'} words: {lines[e]} lines ({100 * lines[e] / n:.1f}%), "
                  f"{100 * words[e] / total:.1f}% of the words")
            low = e
        print(f"words past a line's first {WINDOW}: {cut} ({100 * cut / total:.1f}% of the words)")

    def length(line_id: int):
        row = con.execute("SELECT content FROM line WHERE id = ?", (line_id,)).fetchone()
        return len(clean(row[0] or "").split()) if row else None

    def show(name: str, ids: list[int]) -> None:
        ws = sorted(w for w in map(length, ids) if w is not None)
        if ws:
            share = lambda k: f"{k} ({100 * k / len(ws):.0f}%)"
            print(f"{name}: {len(ws)} lines, median {ws[len(ws) // 2]} words; over {TRAINED}: {share(sum(w > TRAINED for w in ws))}; "
                  f"over {WINDOW}: {share(sum(w > WINDOW for w in ws))}; {SHORT} or fewer: {share(sum(w <= SHORT for w in ws))}")

    for item in args.grades:
        name, path = item.split("=", 1)
        rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
        if rows and "from" in rows[0]:
            show(f"{name}, sources the judges knew", [int(r["lineId"]) for r in rows if r["from"] == "knowledge" and int(r["grade"]) >= 2])
            rows = [r for r in rows if r["from"] != "knowledge"]
        show(f"{name}, graded 2 or 3", [int(r["lineId"]) for r in rows if int(r["grade"]) >= 2])
        show(f"{name}, graded 0", [int(r["lineId"]) for r in rows if int(r["grade"]) == 0])


if __name__ == "__main__":
    main()

"""The searches of a harness run as pages a judge reads (README.md): each search's first lines, with enough of their
text to grade them.

    python judge/make_batches.py build/judge/run.jsonl build/judge/batches [--check build/judge/check.jsonl]

run.jsonl is eval-100/harness/Eval.kt's output for pick_searches.py's queries.tsv: the page is its `hybrid` list, what
the app shows. A line of more than 70 words is shown by its beginning and the part nearest the query's words; niqqud is
left out. --check adds two searches already judged by hand to every batch, to measure each judge against them
(agree.py). --lines shows the lines of a tsv (qid, lineId) instead of the page's first ones: a calibration against
lines users graded. The batches hold the searches and the library's text: they stay outside the repository.
"""

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pardes_semantic.clean import clean  # noqa: E402

WHOLE, HEAD, WINDOW = 70, 10, 55  # words: a line shown whole, and the two parts of a longer one


def plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", clean(text or ""))
    return "".join(c for c in text if not unicodedata.combining(c))


def key(word: str) -> str:
    return re.sub(r"[^\w]", "", word)


def page(con: sqlite3.Connection, search: dict, lines: list[int]) -> str:
    wanted = {key(w) for w in plain(search["query"]).split() if len(key(w)) >= 3}
    out = [f"## {search['id']}. {search['query']}"]
    for place, line_id in enumerate(lines, 1):
        content, ref, title = con.execute(
            "SELECT l.content, l.heRef, b.title FROM line l JOIN book b ON b.id = l.bookId WHERE l.id = ?",
            (line_id,)).fetchone()
        words = plain(content).split()
        if len(words) <= WHOLE:
            text = " ".join(words)
        else:
            hits = [i for i, w in enumerate(words) if any(q in key(w) for q in wanted)]
            best = max(hits, key=lambda i: sum(1 for j in hits if i <= j < i + 40), default=0)
            start = max(0, best - 10)
            text = " ".join(words[:HEAD]) + " […] " + " ".join(words[start:start + WINDOW]) + " […]"
        out.append(f"{place}. [{line_id}] {ref or title} ({len(words)}w): {text}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--db", type=Path, default=ROOT / "build/library/seforim.db")
    ap.add_argument("--size", type=int, default=25, help="searches a batch")
    ap.add_argument("--top", type=int, default=6, help="lines of the page shown for a search")
    ap.add_argument("--check", type=Path, help="searches already judged (a run.jsonl), two of them added to every batch")
    ap.add_argument("--lines", type=Path, help="a tsv (qid, lineId) of the lines to show instead of the page's first ones")
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    runs = [json.loads(line) for line in args.run.open(encoding="utf-8")]
    shown = None
    if args.lines:
        shown = {}
        for row in args.lines.read_text("utf-8").splitlines()[1:]:
            search, line_id = row.split("\t")[:2]
            shown.setdefault(search, []).append(int(line_id))
    check = [json.loads(line) for line in args.check.open(encoding="utf-8")] if args.check else []
    args.out.mkdir(parents=True, exist_ok=True)
    batches = 0
    for start in range(0, len(runs), args.size):
        batches += 1
        part = list(runs[start:start + args.size])
        for k in range(2 if check else 0):
            part.insert((k + 1) * len(part) // 3, check[(2 * (batches - 1) + k) % len(check)])
        pages = [page(con, s, shown[str(s["id"])] if shown else [h["lineId"] for h in s["hybrid"][:args.top]])
                 for s in part]
        (args.out / f"batch-{batches:02d}.md").write_text("\n\n".join(pages) + "\n", "utf-8")
    print(f"{batches} batches of {args.size} -> {args.out}")


if __name__ == "__main__":
    main()

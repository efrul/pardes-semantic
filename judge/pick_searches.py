"""The feedback's searches a judge grades (README.md): Pardes's own, with no grade of a user, not a query of an
evaluation, made over the whole library (no book, category or chapter chosen), of 2 to 8 words.

    python judge/pick_searches.py --history build/feedback/history.tsv --graded build/feedback/queries.tsv --out build/judge

history.tsv and queries.tsv are feedback/collect.py's, outside the repository. Writes there too: queries.tsv, the
searches as eval-100/harness/Eval.kt reads them (id, group, query, expected), numbered in a shuffled order (--seed);
and key.tsv, each search's id in this repository's grades.tsv (j-0001 is search 1) beside its query. A grade names a
search only by that id: the key stays with the feedback's download, and the numbers hold only for the download it was
drawn from.
"""

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALUATIONS = ("eval-real/queries.tsv", "eval-v1/queries.tsv", "eval-100/queries.tsv")
WORDS = (2, 8)


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))


def qid(number: int) -> str:
    return f"j-{number:04d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, required=True, help="feedback/collect.py's history.tsv")
    ap.add_argument("--graded", type=Path, required=True, help="feedback/collect.py's queries.tsv: the searches users graded")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=5785)
    args = ap.parse_args()

    # A row whose columns a tab in the query shifted has no time in its place
    history = [r for r in rows(args.history) if (r.get("lastAt") or "").isdigit()]
    graded = {r["query"] for r in rows(args.graded)}
    evaluated = {(r.get("query") or "").strip() for path in EVALUATIONS for r in rows(ROOT / path)}
    scoped = {r["query"] for r in history if r["bookId"] or r["categoryId"] or r["tocId"]}
    own = {r["query"] for r in history if (r["source"] or "pardes") == "pardes"}
    picked = sorted(own - graded - evaluated - scoped)
    picked = [q for q in picked if WORDS[0] <= len(q.split()) <= WORDS[1] and "\t" not in q and '"' not in q]
    random.Random(args.seed).shuffle(picked)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "queries.tsv").open("w", encoding="utf-8") as f:
        f.write("id\tgroup\tquery\texpected\n")
        for number, query in enumerate(picked, 1):
            f.write(f"{number}\tjudge\t{query}\t\n")
    with (args.out / "key.tsv").open("w", encoding="utf-8") as f:
        f.write("qid\tquery\n")
        for number, query in enumerate(picked, 1):
            f.write(f"{qid(number)}\t{query}\n")
    print(f"{len(own)} searches of Pardes; {len(picked)} to judge -> {args.out / 'queries.tsv'}, {args.out / 'key.tsv'}")


if __name__ == "__main__":
    main()

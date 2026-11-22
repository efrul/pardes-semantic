"""The judges' files as this repository keeps them (README.md): grades.tsv and status.tsv, each search named by its id
in pick_searches.py's key (j-0001 is search 1), with no query, no text and none of the judges' notes.

    python judge/merge_grades.py --hand build/demo/judged.tsv build/demo/notes.tsv \
        --grades build/judge/out/grades-*.tsv --notes build/judge/out/notes-*.tsv --out judge

A judge writes two files a batch (INSTRUCTIONS.md): grades (qid, lineId, grade, from) and notes (qid, status, note),
the search by its number. --hand gives the same two files of the searches judged by hand: they are the check searches
that make_batches.py puts in every batch, so their grades come from --hand only, and what the judges gave them is
measured with agree.py and left out here. A row that is not a grade (a line id or a grade that is not a number, a
status that is not one of the four) is counted and skipped.
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from pick_searches import qid

STATUSES = ("ok", "none", "unclear", "noise")


def table(path: Path) -> list[list[str]]:
    return [row for row in csv.reader(path.open(encoding="utf-8"), delimiter="\t") if row and row[0] != "qid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", type=Path, nargs=2, metavar=("GRADES", "NOTES"), help="the searches judged by hand")
    ap.add_argument("--grades", type=Path, nargs="+", required=True)
    ap.add_argument("--notes", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    grades, status, skipped = {}, {}, Counter()

    def read(grade_file: Path, note_file: Path | None, judge: str, protected: set[str]) -> None:
        for row in table(grade_file):
            if len(row) < 3 or not row[0].isdigit() or not row[1].strip().isdigit() or row[2].strip() not in list("0123"):
                skipped["a row that is not a grade"] += 1
            elif row[0] in protected:
                skipped["a judge's grade of a check search"] += 1
            else:
                source = row[3].strip() if len(row) > 3 and row[3].strip() in ("page", "knowledge") else "page"
                grades[(int(row[0]), int(row[1]))] = (int(row[2]), source, judge)
        for row in table(note_file) if note_file else []:
            if len(row) < 2 or not row[0].isdigit() or row[1].strip() not in STATUSES:
                skipped["a row that is not a status"] += 1
            elif row[0] not in protected:
                status[int(row[0])] = (row[1].strip(), judge)

    hand = set()
    if args.hand:
        read(args.hand[0], args.hand[1], "hand", set())
        hand = {str(q) for q in status}
    notes = {re.sub(r"\D", "", p.stem): p for p in args.notes}
    for path in sorted(args.grades):
        batch = re.sub(r"\D", "", path.stem)
        read(path, notes.get(batch), f"batch-{batch}", hand)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "grades.tsv").open("w", encoding="utf-8") as f:
        f.write("qid\tlineId\tgrade\tfrom\tjudge\n")
        for (search, line), (grade, source, judge) in sorted(grades.items(), key=lambda kv: (kv[0][0], kv[1][1] != "page", kv[0][1])):
            f.write(f"{qid(search)}\t{line}\t{grade}\t{source}\t{judge}\n")
    with (args.out / "status.tsv").open("w", encoding="utf-8") as f:
        f.write("qid\tstatus\tjudge\n")
        for search, (state, judge) in sorted(status.items()):
            f.write(f"{qid(search)}\t{state}\t{judge}\n")
    print(f"{len(status)} searches: {dict(Counter(s for s, _ in status.values()))}")
    print(f"{len(grades)} grades: {dict(sorted(Counter(g for g, _, _ in grades.values()).items()))}, "
          f"{sum(s == 'knowledge' for _, s, _ in grades.values())} of lines added from knowledge")
    if skipped:
        print("skipped:", dict(skipped))


if __name__ == "__main__":
    main()

"""How a judge's grades agree with a key's on whether a line is good (2 or 3), with Cohen's kappa: against the grades
users gave (a calibration), or against the check searches judged by hand (README.md).

    python judge/agree.py KEY.tsv GRADES.tsv [GRADES.tsv ...]

Every file: qid, lineId, grade, tab-separated, a header line or none. Counts only are printed.
"""

import csv
import sys
from collections import Counter


def load(path: str) -> dict[tuple[str, str], int]:
    out = {}
    for row in csv.reader(open(path, encoding="utf-8"), delimiter="\t"):
        if row and row[0] != "qid" and len(row) >= 3 and row[2].strip().isdigit():
            out[(row[0], row[1])] = int(row[2])
    return out


def main() -> None:
    key = load(sys.argv[1])
    judge = {}
    for path in sys.argv[2:]:
        judge |= load(path)
    both = [(key[k], judge[k]) for k in key if k in judge]
    n = len(both)
    if not n:
        sys.exit("no line in common")
    agree = sum((a >= 2) == (b >= 2) for a, b in both) / n
    by_key, by_judge = sum(a >= 2 for a, _ in both) / n, sum(b >= 2 for _, b in both) / n
    chance = by_key * by_judge + (1 - by_key) * (1 - by_judge)
    print(f"{n} lines in common of {len(key)} in the key | agree whether good: {agree:.0%}, "
          f"kappa {(agree - chance) / (1 - chance):.2f}")
    print(f"good by the key {by_key:.0%}, by the judge {by_judge:.0%}")
    print("key grade -> judge's grades:",
          {g: dict(sorted(Counter(b for a, b in both if a == g).items())) for g in sorted({a for a, _ in both})})
    print("good by the key only:", sum(a >= 2 > b for a, b in both), "| good by the judge only:",
          sum(b >= 2 > a for a, b in both))


if __name__ == "__main__":
    main()

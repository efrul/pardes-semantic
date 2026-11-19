"""The evaluations judged by grades, measured on the pool (README.md): eval-real's searches, eval-v1's batch 2 and the
feedback's searches graded by the app's users (../feedback/collect.py's searches.json, merged into pool.json). Each
model orders only the search's judged lines, as results-run3.md's tables of the real searches do: the lines nobody
graded are left out, so the order among the judged lines is measured, not what the model finds in the library.

    python judged_measure.py ranks-base.json ranks-teacher.json ranks-student.json ranks-tuned.json \
        --feedback build/feedback/grades.tsv [--md]

A model is named by its file (ranks-<name>.json, ../pool_rank.py's). Per evaluation and model: the searches whose first
judged line is good (graded 2 or 3), with a good line in the first three, and with the source (3) in the first three.
Each pair of models is compared search by search on the first good line, as ../eval-crfh/measure.py compares them: the
searches only one of them gets, and whether the difference passes twice its noise (2 * sqrt(b + c), McNemar's).
"""

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOOD, SOURCE = 2, 3


def grades_of(path: Path) -> dict[str, dict[int, int]]:
    """A grades.tsv (qid, lineId, grade, ...) as {qid: {line: grade}}."""
    graded = {}
    for r in csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"):
        graded.setdefault(r["qid"], {})[int(r["lineId"])] = int(r["grade"])
    return graded


def judged_order(ranks: dict, key: str, graded: dict[int, int]) -> list[int] | None:
    """The grades of the search's judged lines in the model's order, or None when it ranked none of them."""
    r = ranks.get(key)
    if r is None:
        return None
    order = [graded[line] for line in r["ids"] if line in graded]
    return order or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ranks", nargs="+", type=Path)
    ap.add_argument("--feedback", type=Path, help="collect.py's grades.tsv of the feedback's searches in pool.json")
    ap.add_argument("--md", action="store_true", help="markdown tables")
    args = ap.parse_args()
    evaluations = {"real": grades_of(HERE.parent / "eval-real" / "grades.tsv"),
                   "v1": grades_of(HERE.parent / "eval-v1" / "grades.tsv")}
    if args.feedback:
        evaluations["feedback"] = grades_of(args.feedback)
    models = {path.stem.removeprefix("ranks-"): json.loads(path.read_text("utf-8")) for path in args.ranks}

    # {evaluation: {model: {qid: (first good, good in three, source in three)}}}
    found = {}
    for name, graded in evaluations.items():
        for model, ranks in models.items():
            for qid, lines in graded.items():
                order = judged_order(ranks, f"{name}/{qid}", lines)
                if order is not None:
                    found.setdefault(name, {}).setdefault(model, {})[qid] = (
                        order[0] >= GOOD, any(g >= GOOD for g in order[:3]), any(g >= SOURCE for g in order[:3]))

    head = ["evaluation", "model", "searches", "first good", "good in 3", "source in 3"]
    rows, pairs = [], []
    for name, by_model in found.items():
        for model, of in by_model.items():
            rows.append([name, model, len(of)] + [sum(1 for v in of.values() if v[i]) for i in range(3)])
        for a, b in combinations(by_model, 2):
            common = by_model[a].keys() & by_model[b].keys()
            only_a = sum(1 for q in common if by_model[a][q][0] and not by_model[b][q][0])
            only_b = sum(1 for q in common if by_model[b][q][0] and not by_model[a][q][0])
            bar = 2 * math.sqrt(only_a + only_b)
            verdict = ("within the noise" if abs(only_a - only_b) < bar or only_a == only_b
                       else f"{a} better" if only_a > only_b else f"{b} better")
            pairs.append([name, f"{a} / {b}", only_a, only_b, f"{bar:.1f}", verdict])
    pair_head = ["evaluation", "models", "first good: only the first", "only the second", "noise bar", "verdict"]
    if args.md:
        for h, rs in ((head, rows), (pair_head, pairs)):
            print("| " + " | ".join(h) + " |\n|" + "---|" * len(h))
            for r in rs:
                print("| " + " | ".join(map(str, r)) + " |")
            print()
    else:
        for h, rs in ((head, rows), (pair_head, pairs)):
            print("\t".join(h))
            for r in rs:
                print("\t".join(map(str, r)))
            print()


if __name__ == "__main__":
    main()

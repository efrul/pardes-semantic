"""The judged evaluation's searches by the training searches near them (README.md). The app's users ask a need again
in other words, so a search of the judged evaluation often has training searches beside it, and a model that learnt
the pairs is measured there on what it was shown.

    python judge/near_training.py --db build/library/seforim.db --pairs build/judged/pairs.jsonl \
        --searches build/judged/searches.json --grades build/judged/grades.tsv ranks-a.json ranks-b.json

--pairs, --searches and --grades are build_pairs.py's; a model is named by its file (ranks-<name>.json, ../pool_rank.py's
of the pool that holds the judged searches). A training search is near an evaluation search when a good line (2 or 3) of
the evaluation search is the positive of one of its pairs, or when the two share half their words or more (the words in
common, of the words of both). Per group (no training search near, one or two, three or more) and model: the searches
whose first judged line is good, with a good line in the first three, and with the source in the first three, as
../eval-pool/judged_measure.py counts them; and each two models search by search on the first.
"""

import argparse
import csv
import json
import math
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train-v2"))
from build_pairs import letters  # noqa: E402
from build_train_pairs import MAX_PASSAGE_WORDS  # noqa: E402
from pardes_semantic.clean import clean  # noqa: E402

GOOD, SOURCE = 2, 3
GROUPS = (("no training search near", 0, 0), ("one or two", 1, 2), ("three or more", 3, None), ("all", 0, None))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ranks", nargs="*", type=Path)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True, help="build_pairs.py's pairs.jsonl, with all the pairs")
    ap.add_argument("--searches", type=Path, required=True)
    ap.add_argument("--grades", type=Path, required=True)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    by_positive, words_of = {}, {}
    for line in args.pairs.open(encoding="utf-8"):
        pair = json.loads(line)
        by_positive.setdefault(pair["positive"], set()).add(pair["anchor"])
        words_of[pair["anchor"]] = set(letters(pair["anchor"]).split())
    grades = {}
    for r in csv.DictReader(args.grades.open(encoding="utf-8"), delimiter="\t"):
        grades.setdefault(r["qid"], {})[int(r["lineId"])] = int(r["grade"])
    searches = {s["key"]: s["query"] for s in json.loads(args.searches.read_text("utf-8"))["searches"]}

    def passage(line_id: int) -> str:
        (content,) = con.execute("SELECT content FROM line WHERE id = ?", (line_id,)).fetchone()
        return " ".join(clean(content or "").split()[:MAX_PASSAGE_WORDS])

    near = {}
    for key, query in searches.items():
        words = set(letters(query).split())
        beside = {a for a, other in words_of.items() if other and len(words & other) / len(words | other) >= 0.5}
        for line_id, grade in grades[key].items():
            if grade >= GOOD:
                beside |= by_positive.get(passage(line_id), set())
        near[key] = len(beside)

    models = {path.stem.removeprefix("ranks-"): json.loads(path.read_text("utf-8")) for path in args.ranks}
    print("group\tmodel\tsearches\tfirst good\tgood in 3\tsource in 3")
    pairs = []
    for label, least, most in GROUPS:
        keys = [k for k, n in near.items() if n >= least and (most is None or n <= most)]
        if not models:
            print(f"{label}\t\t{len(keys)}")
        found = {}
        for name, ranks in models.items():
            found[name] = {}
            for key in keys:
                order = [grades[key][line] for line in ranks[key]["ids"] if line in grades[key]]
                if order:
                    found[name][key] = (order[0] >= GOOD, any(g >= GOOD for g in order[:3]),
                                        any(g >= SOURCE for g in order[:3]))
            of = found[name].values()
            print(f"{label}\t{name}\t{len(of)}\t" + "\t".join(str(sum(v[i] for v in of)) for i in range(3)))
        for a, b in combinations(models, 2):
            common = found[a].keys() & found[b].keys()
            only_a = sum(1 for k in common if found[a][k][0] and not found[b][k][0])
            only_b = sum(1 for k in common if found[b][k][0] and not found[a][k][0])
            pairs.append(f"{label}\t{a} / {b}\t{only_a}\t{only_b}\t{2 * math.sqrt(only_a + only_b):.1f}")
    if pairs:
        print("\ngroup\tmodels\tfirst good: only the first\tonly the second\tnoise bar")
        print("\n".join(pairs))


if __name__ == "__main__":
    main()

"""The judges' grades as training pairs and as an evaluation, as feedback/collect.py makes them of the users' grades
(README.md): the same pairs.jsonl that train_model.py --negatives reads, the same hold-out of what the evaluations judge.

    python judge/build_pairs.py --db build/library/seforim.db --key build/judge/key.tsv --out build/judged \
        [--release-old-benchmark]

grades.tsv and status.tsv are this folder's; --key is pick_searches.py's key.tsv, which gives each id its query and
stays outside the repository, as --out does: pairs.jsonl holds the searches and the library's text.

- A search of status `ok` with a good line (2 or 3) is used; `unclear`, `noise` and `none` give nothing.
- A third of the searches (--eval-share), chosen by their words alone so that a search stays on its side at every
  build, is an evaluation: searches.json (pool_spec.py's form, to merge into pool.json) and grades.tsv (the --feedback
  of eval-pool/judged_measure.py). They give no pair.
- Of the others, each good line is a positive, and the lines graded 0 are the negatives. A line graded 1 is neither:
  against the users' grades, most of the lines a judge gave 1 were good in the users' eyes.
- A line the evaluations hold out (feedback/collect.py's HeldOut) is left out, and a search whose query an evaluation
  asks. --release-old-benchmark lets the old benchmark's books (the Shulchan Arukh, Shabbat, Bava Kamma, Bava Metzia,
  Sanhedrin) and the Rambam halachot it asks with be learnt from: the outside set's tractates (Berakhot among them) and
  what the judged evaluations hold, with its links and neighbours, stay out.
"""

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train-v2"))
sys.path.insert(0, str(ROOT / "feedback"))
from build_pairs import letters  # noqa: E402
from build_train_pairs import MAX_PASSAGE_WORDS  # noqa: E402
from collect import HeldOut  # noqa: E402
from pardes_semantic.clean import clean  # noqa: E402
from sample_targets import held_book_texts, outside_set_books  # noqa: E402

GOOD = 2


def table(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))


def in_evaluation(words: str, share: float) -> bool:
    return int(hashlib.sha1(words.encode()).hexdigest(), 16) % 1000 < share * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True, help="pick_searches.py's key.tsv (qid, query)")
    ap.add_argument("--grades", type=Path, default=HERE / "grades.tsv")
    ap.add_argument("--status", type=Path, default=HERE / "status.tsv")
    ap.add_argument("--eval-share", type=float, default=1 / 3)
    ap.add_argument("--release-old-benchmark", action="store_true",
                    help="the old benchmark's books and the Rambam halachot it asks with may be learnt from")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    query = {r["qid"]: r["query"] for r in table(args.key)}
    status = {r["qid"]: r["status"] for r in table(args.status)}
    grades, source = {}, {}
    for r in table(args.grades):
        grades.setdefault(r["qid"], {})[int(r["lineId"])] = int(r["grade"])
        source[(r["qid"], int(r["lineId"]))] = r["from"]
    missing = sorted(set(grades) - set(query))
    if missing:
        raise SystemExit(f"{len(missing)} searches of {args.grades} are not in {args.key} (the first: {missing[0]}): "
                         "the key of another download?")

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    ids = sorted({line for lines in grades.values() for line in lines})
    lines = {}
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for line_id, book, index, ref, content in con.execute(
                f"SELECT id, bookId, lineIndex, heRef, content FROM line WHERE id IN ({','.join('?' * len(chunk))})",
                chunk):
            lines[line_id] = (book, index, ref, content or "")
    held = HeldOut(con)
    if args.release_old_benchmark:
        held.books = outside_set_books(con)
        held.lines = set()
        held.texts = held_book_texts(con, held.books)

    def passage(line_id: int) -> str:
        return " ".join(clean(lines[line_id][3]).split()[:MAX_PASSAGE_WORDS])

    dropped, pairs, spec, judged = Counter(), [], [], []
    seen = set()
    for q in sorted(grades):
        if status.get(q, "ok") != "ok":
            dropped[f"status {status[q]}"] += 1
            continue
        words = letters(query[q])
        if words in seen:
            dropped["the same words as an earlier search"] += 1
            continue
        seen.add(words)
        if held.query(query[q]):
            dropped["an evaluation's query"] += 1
            continue
        graded = {line: grade for line, grade in grades[q].items() if line in lines}
        dropped["a line not in this library"] += len(grades[q]) - len(graded)
        if not any(grade >= GOOD for grade in graded.values()):
            dropped["no good line"] += 1
            continue
        if in_evaluation(words, args.eval_share):
            spec.append({"key": f"judged/{q}", "query": query[q], "candidates": sorted(graded)})
            judged += [(q, line, grade, source[(q, line)]) for line, grade in graded.items()]
            continue

        def usable(line_id: int) -> bool:
            if held.line(line_id, lines[line_id]):
                dropped["a line an evaluation holds out"] += 1
                return False
            return True

        good = [line for line, grade in graded.items() if grade >= GOOD and usable(line)]
        bad = [line for line, grade in graded.items() if grade == 0 and usable(line)]
        dropped["a line graded 1, neither positive nor negative"] += sum(grade == 1 for grade in graded.values())
        if not good:
            dropped["every good line held out"] += 1
            continue
        for line in good:
            pairs.append({"anchor": query[q], "positive": passage(line),
                          "type": f"judged-{source[(q, line)]}-{graded[line]}", "negatives": [passage(b) for b in bad]})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "searches.json").write_text(json.dumps({"searches": spec}, ensure_ascii=False) + "\n", "utf-8")
    with (args.out / "grades.tsv").open("w", encoding="utf-8") as f:
        f.write("qid\tlineId\tgrade\tfrom\n")
        for q, line, grade, where in judged:
            f.write(f"judged/{q}\t{line}\t{grade}\t{where}\n")
    with (args.out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    negatives = {}
    for p in pairs:
        negatives[p["anchor"]] = len(p["negatives"])
    print(f"{len(grades)} searches judged: {len(spec)} for the evaluation ({len(judged)} graded lines) -> "
          f"{args.out / 'searches.json'}; {len(negatives)} for training: {len(pairs)} pairs "
          f"({sum('knowledge' in p['type'] for p in pairs)} of lines added from knowledge), "
          f"{sum(negatives.values())} negatives -> {args.out / 'pairs.jsonl'}")
    print("left out:", dict(dropped))


if __name__ == "__main__":
    main()

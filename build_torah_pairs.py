"""Training pairs from Torah-pairs (TRAINING_PLAN.md, רישוי): a question, or the topic of a responsum, as the query, and
the line of the library that answers it as the passage to find. Its queries are a reader's own way of asking ("מה
הדין..."), which the links' pairs lack, and each points at a line of ours by its reference.

    python build_torah_pairs.py --db seforim.db --dir build/torah-pairs --out build/train/torah-pairs.jsonl \
        [--take torahqa.clean=40000,shut_passages=10000] [--test 3000] [--hold pool.json ...] [--seed 5811]

The files themselves are not in this repository and are not passed on (CLAUDE.md): they are read from --dir, and what
this writes stays under build/. Only the two files the user chose are read by default (2026-11-24): the questions with
their answers (the question alone is the query) and the responsa's topics.

A row's passage is found by its Hebrew reference, compared with ours without spaces and commas (the files' book ids are
Otzaria's own); a questions' row that names the best line of its passage is given that line. A line gives at most
PER_LINE queries, a query is used once.

Nothing an evaluation measures is touched, as in build_chavruta_pairs.py: no line of the benchmark's books or of
crfh1672/search-queries' tractates, none an evaluation judges or targets, nor linked to one or near one, none near a
dev question's target, no copy of a held book's line; and with --hold, no line a pool's searches rank (the judged
searches, the feedback). --test keeps that many pairs out, whole: their lines, the lines near them and their queries
are not learnt from, and they are written beside --out (-test.jsonl) to measure a model on questions it did not see.
"""

import argparse
import csv
import json
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "train-v2"))
from build_pairs import DEV_BATCHES, read_tsv  # noqa: E402
from build_train_pairs import MAX_PASSAGE_WORDS, MIN_WORDS, benchmark_held  # noqa: E402
from pardes_semantic.clean import clean  # noqa: E402
from sample_targets import NEAR, held_book_texts, held_out, neighbors, outside_set_books  # noqa: E402

PER_LINE = 2
QUERY_WORDS = (3, 60)
# The field of a row that is its query, and the name its pairs carry
FILES = {"torahqa.clean": ("question", "TORAH-QA"), "shut_passages": ("query", "TORAH-SHUT"),
         "micropedia": ("query", "TORAH-MICROPEDIA"), "et": ("query", "TORAH-ET")}
CONFIDENT = {"high", "exact", "strong"}
MARKS = re.compile(r"[֑-ׇ]")


def ref_key(ref: str | None) -> str:
    return re.sub(r"[ ,]", "", ref or "")


def plain(text: str) -> str:
    return " ".join(MARKS.sub("", text or "").replace("־", " ").split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--dir", type=Path, required=True, help="the folder of Torah-pairs' .jsonl files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--take", default="torahqa.clean=40000,shut_passages=10000",
                    help="comma separated: <file>=<pairs at most>")
    ap.add_argument("--test", type=int, default=3000, help="pairs kept out whole, written beside --out")
    ap.add_argument("--hold", type=Path, action="append", default=[],
                    help="a pool spec (eval-pool/pool_spec.py's): no line its searches rank; repeatable")
    ap.add_argument("--hold-grades", type=Path, action="append", default=[],
                    help="a grades.tsv (qid, lineId, ...): no line it grades; repeatable")
    ap.add_argument("--seed", type=int, default=5811)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    take = {name: int(n) for name, n in (part.split("=") for part in args.take.split(","))}
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    held_books, held_lines = benchmark_held(con)
    held_books |= outside_set_books(con)
    held_texts = held_book_texts(con, held_books)
    judged, patterns, quotes = held_out(con)
    for path in args.hold_grades:
        judged |= {int(r["lineId"]) for r in csv.DictReader(path.open(encoding="utf-8"), delimiter="\t",
                                                           quoting=csv.QUOTE_NONE)}
    linked, near = neighbors(con, judged)
    dev = {int(q["target_line"]) for q in read_tsv(HERE / "train-v2" / "questions.tsv") if q["batch"] in DEV_BATCHES}
    dev_linked, dev_near = neighbors(con, dev)
    plain_quotes = [q for q in quotes if q]
    ranked = set()
    for path in args.hold:
        for s in json.loads(path.read_text("utf-8"))["searches"]:
            ranked.update(s["candidates"])
    ours = {}
    for line_id, ref in con.execute("SELECT id, heRef FROM line WHERE heRef IS NOT NULL"):
        ours.setdefault(ref_key(ref), line_id)
    print(f"{len(held_books)} books held out, {len(linked)} lines judged or linked to one, {len(ranked)} lines a pool "
          f"ranks; {len(ours)} references of ours", file=sys.stderr)

    def line(lid):
        return con.execute("SELECT bookId, lineIndex, heRef, content FROM line WHERE id = ?", (lid,)).fetchone()

    def touched(lid, row):
        book, index, ref, content = row
        return (lid in held_lines or book in held_books or lid in linked or (book, index) in near
                or lid in dev_linked or (book, index) in dev_near or lid in ranked)

    usable = {}                # file -> [(query, line id, book id, line index, passage)]
    why = Counter()
    for name in take:
        field, _ = FILES[name]
        rows = []
        with (args.dir / f"{name}.jsonl").open(encoding="utf-8") as f:
            for raw in f:
                d = json.loads(raw)
                p = d.get("passage") if isinstance(d.get("passage"), dict) else {}
                lid = ours.get(ref_key(p.get("he_ref")))
                if lid is None:
                    why["not a line of ours"] += 1
                    continue
                if p.get("confidence") not in CONFIDENT:
                    why["a weak match"] += 1
                    continue
                row = line(lid)
                best, first = p.get("best_line"), p.get("line_index")
                if isinstance(best, int) and isinstance(first, int) and best != first:   # the passage's best line
                    other = con.execute("SELECT id FROM line WHERE bookId = ? AND lineIndex = ?",
                                        (row[0], row[1] + best - first)).fetchone()
                    if other:
                        lid, row = other[0], line(other[0])
                query = plain(clean(d.get(field) or ""))
                if not QUERY_WORDS[0] <= len(query.split()) <= QUERY_WORDS[1]:
                    why["a query too short or too long"] += 1
                    continue
                if touched(lid, row) or (row[2] and any(pt.search(row[2]) for pt in patterns)):
                    why["near what an evaluation measures"] += 1
                    continue
                text = clean(row[3])
                if text in held_texts:
                    why["a copy of a held book's line"] += 1
                    continue
                bare = " ".join(re.sub(r"[^א-ת ]", " ", MARKS.sub("", row[3].replace("־", " "))).split())
                if any(q in bare for q in plain_quotes):
                    why["holds a quote's source"] += 1
                    continue
                passage = " ".join(text.split()[:MAX_PASSAGE_WORDS])
                if len(passage.split()) < MIN_WORDS:
                    why["a line too short"] += 1
                    continue
                rows.append((query, lid, row[0], row[1], passage))
        usable[name] = rows
        print(f"{name}: {len(rows)} usable rows", file=sys.stderr)
    print("left out: " + ", ".join(f"{k} {v}" for k, v in why.most_common()), file=sys.stderr)

    # The test pairs first: a query once, a line once, from every file by its share of --take
    everything = [(name, r) for name, rows in usable.items() for r in rows]
    rng.shuffle(everything)
    total = sum(take.values())
    test, test_queries, test_lines, test_near = [], set(), set(), set()
    quota = {name: round(args.test * n / total) for name, n in take.items()}
    for name, (query, lid, book, index, passage) in everything:
        if quota[name] <= 0 or query in test_queries or lid in test_lines or (book, index) in test_near:
            continue
        quota[name] -= 1
        test.append({"anchor": query, "line": lid, "type": FILES[name][1]})
        test_queries.add(query)
        test_lines.add(lid)
        test_near |= {(book, index + d) for d in range(-NEAR, NEAR + 1)}

    pairs, used = [], set()
    for name, rows in usable.items():
        by_line = defaultdict(list)
        for query, lid, book, index, passage in rows:
            if query in test_queries or query in used or (book, index) in test_near:
                continue
            by_line[lid].append((query, passage))
        got = []
        for lid in sorted(by_line):
            for query, passage in rng.sample(by_line[lid], min(PER_LINE, len(by_line[lid]))):
                if query not in used:
                    used.add(query)
                    got.append({"anchor": query, "positive": passage, "type": FILES[name][1], "line": lid})
        rng.shuffle(got)
        pairs += got[:take[name]]
        print(f"{name}: {len(by_line)} lines, {len(got)} pairs, {min(len(got), take[name])} taken", file=sys.stderr)
    rng.shuffle(pairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    test_path = args.out.with_name(args.out.stem + "-test.jsonl")
    with test_path.open("w", encoding="utf-8") as f:
        for p in test:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(pairs)} pairs -> {args.out}; {len(test)} kept out -> {test_path}")


if __name__ == "__main__":
    main()

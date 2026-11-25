"""Hard negatives for pairs whose passage is a line of the library: the lines the pack's model puts first for a pair's
query over the whole library, of which a reranker keeps only those that do not answer it.

    python mine_hard_negatives.py queries --pairs build/train/torah-pairs.jsonl --out build/train/mine-queries.jsonl
    java -cp "$(cat build/harness/classpath.txt)" MineKt seforim.db build/train/mine-queries.jsonl build/train/mined.jsonl 100 4
    python mine_hard_negatives.py candidates --db seforim.db --pairs build/train/torah-pairs.jsonl \
        --mined build/train/mined.jsonl --tag t --out build/train/denoise-t.json [--hold ...] [--hold-lines test.jsonl ...]
    python kaggle_run.py push pool --as pooln --spec <the specs merged> --models tune3/tuned@merge+strip \
        --rerank Horizon-Labs/multilingual-reranker-small --rerank-top 5 ...
    python mine_hard_negatives.py apply --db seforim.db --pairs build/train/torah-pairs.jsonl --tag t \
        --spec build/train/denoise-t.json --scores ranks-...-multilingual-reranker-small.json --out build/train/torah-pairs-neg.jsonl

Why a reranker: the library says the same thing in many places (the Gemara, the Rambam that codifies it, its
commentaries, a later book that quotes them, the Mishnah in two editions), and a pair names one of them. Of 24 lines the
model put first and below the pair's own line, read by hand (eval-pool/results-torah-pairs.md), 9 answered the question
and 7 more were close to it: taken as negatives, they would teach the model to push answers away. The positive-aware
rule alone (NV-Retriever, Moreira et al. 2024: a line below 95% of the pair's own score) does not tell them apart,
since the pair's own line is often not the best answer. A cross-encoder reads the query with each line, as RocketQA
(Qu et al. 2021) denoises its hard negatives.

candidates: for each pair, its own line and the first K lines the model found that may be negatives at all:
- not the passage, not within NEAR_BOOK lines of it in its book, not linked to it or to a line linked to it (a
  commentary on it, another on the same line, the same law in a code and its commentaries); a passage with more than
  SECOND_HOP links gets none, its parallels being too many to tell;
- no line that holds a run of RUN words of the passage (a quote of it, the same text elsewhere);
- nothing an evaluation measures (build_torah_pairs.py's Guard), no line of a kept-out test nor near one (--hold-lines:
  a model pushed away from a test's line would be measured unfairly on it), long enough to be a passage;
- for a statement of Mikropedia Talmudit's (TORAH-MICROPEDIA), not a line of that book: the statement is one of its
  lines, and its other lines restate laws with their sources.
They are written as a pool spec (eval-pool/pool_spec.py's format), the key "<tag>/<row>", for the pool stage's reranker.

apply: a candidate is a negative when the reranker's score of it is under BELOW and under the pair's own line's by
MARGIN or more. A row gets up to PER_ROW of them, in the model's order (the hardest first); train_model.py --negatives
gives a row its first. The rows keep their order, and a row with none is written as it was.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "train-v2"))
from build_torah_pairs import Guard  # noqa: E402
from build_train_pairs import MAX_PASSAGE_WORDS, MIN_WORDS  # noqa: E402
from pardes_semantic.clean import clean  # noqa: E402
from sample_targets import neighbors  # noqa: E402

K = 4
NEAR_BOOK = 10
SECOND_HOP = 5000  # the lines linked to the passage whose own links are read too, at most (a verse's can be many more)
RUN = 8            # a candidate holding this many words of the passage in a row quotes it
BELOW = 0.3        # the reranker's score (0 to 1) a negative stays under
MARGIN = 0.3       # and how far under the pair's own line's
PER_ROW = 3
MARKS = re.compile(r"[֑-ׇ]")
# The books whose lines are no negative for a type of pair: the book the type's queries come from
OWN_BOOK = {"TORAH-MICROPEDIA": "מיקרופדיה תלמודית"}


def words(text: str) -> list[str]:
    return re.sub(r"[^א-ת ]", " ", MARKS.sub("", text.replace("־", " "))).split()


def runs(text: str) -> set[tuple]:
    w = words(text)
    return {tuple(w[i:i + RUN]) for i in range(len(w) - RUN + 1)}


def queries(args) -> None:
    n = 0
    with args.pairs.open(encoding="utf-8") as f, args.out.open("w", encoding="utf-8") as out:
        for i, raw in enumerate(f):
            row = json.loads(raw)
            out.write(json.dumps({"id": i, "query": row["anchor"], "line": row["line"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} queries -> {args.out}")


def candidates(args) -> None:
    import sqlite3
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = [json.loads(raw) for raw in args.pairs.open(encoding="utf-8")]
    mined = {}
    for raw in args.mined.open(encoding="utf-8"):
        m = json.loads(raw)
        mined[m["id"]] = m
    guard = Guard(con, args.hold, args.hold_grades)
    held = set()
    for path in args.hold_lines:
        lines = {json.loads(raw)["line"] for raw in path.open(encoding="utf-8")}
        held |= neighbors(con, lines)[1]
    own_book = {kind: con.execute("SELECT id FROM book WHERE title = ?", (title,)).fetchone()[0] for kind, title in OWN_BOOK.items()}

    def line(lid):
        return con.execute("SELECT bookId, lineIndex, heRef, content FROM line WHERE id = ?", (lid,)).fetchone()

    searches, why, place = [], Counter(), Counter()
    for i, row in enumerate(rows):
        m = mined.get(i)
        if m is None or "hits" not in m:
            why["not searched"] += 1
            continue
        ids = [h[0] for h in m["hits"]]
        rank = ids.index(row["line"]) + 1 if row["line"] in ids else 0
        place["first" if rank == 1 else "2 to 10" if 1 < rank <= 10 else f"11 to {len(ids)}" if rank else "beyond"] += 1
        own = line(row["line"])
        linked, _ = neighbors(con, {row["line"]})
        if len(linked) > SECOND_HOP:
            why["the passage has too many links to tell its parallels"] += 1
            continue
        linked |= neighbors(con, linked)[0]
        quoted = runs(own[3])
        found = []
        for lid in ids:
            if lid == row["line"] or lid in linked:
                continue
            other = line(lid)
            if other is None:
                continue
            if other[0] == own[0] and abs(other[1] - own[1]) <= NEAR_BOOK:
                why["near the passage in its book"] += 1
                continue
            if (other[0], other[1]) in held:
                why["a line of a kept-out test, or near one"] += 1
                continue
            if other[0] == own_book.get(row.get("type")):
                why["a line of the book the query comes from"] += 1
                continue
            if len(clean(other[3]).split()) < MIN_WORDS:
                continue
            if quoted & runs(other[3]):
                why["quotes the passage"] += 1
                continue
            if guard.left_out(lid, other):
                why["a line an evaluation measures"] += 1
                continue
            found.append(lid)
            if len(found) == args.k:
                break
        if found:
            searches.append({"key": f"{args.tag}/{i}", "query": row["anchor"], "candidates": [row["line"]] + found})
        else:
            why["no line left"] += 1
    args.out.write_text(json.dumps({"searches": searches}, ensure_ascii=False) + "\n", "utf-8")
    print("the passage's place among the lines found: " + ", ".join(f"{k} {v}" for k, v in place.most_common()))
    print(f"{len(searches)} of {len(rows)} rows have candidates, {sum(len(s['candidates']) - 1 for s in searches)} in all")
    print("left out: " + ", ".join(f"{k} {v}" for k, v in why.most_common()))
    print(f"-> {args.out}")


def apply(args) -> None:
    import sqlite3
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = [json.loads(raw) for raw in args.pairs.open(encoding="utf-8")]
    spec = {s["key"]: s["candidates"] for s in json.loads(args.spec.read_text("utf-8"))["searches"]}
    scores = json.loads(args.scores.read_text("utf-8"))
    got, why = Counter(), Counter()
    for key, cands in spec.items():
        tag, _, i = key.partition("/")
        if tag != args.tag:
            continue
        row = rows[int(i)]
        r = scores.get(key)
        if r is None:
            why["not scored"] += 1
            continue
        score = dict(zip(r["ids"], r["scores"]))
        own = score.get(row["line"])
        if own is None:
            why["its own line not scored"] += 1
            continue
        negatives, negative_lines = [], []
        for lid in cands[1:]:              # in the model's order
            s = score.get(lid)
            if s is None:
                continue
            if s >= args.below or s > own - args.margin:
                why["the reranker reads it as an answer, or near one"] += 1
                continue
            text = con.execute("SELECT content FROM line WHERE id = ?", (lid,)).fetchone()[0]
            passage = " ".join(clean(text).split()[:MAX_PASSAGE_WORDS])
            if len(passage.split()) < MIN_WORDS or passage == row["positive"] or passage in negatives:
                continue
            negatives.append(passage)
            negative_lines.append(lid)
            if len(negatives) == args.per_row:
                break
        if negatives:
            row["negatives"] = negatives
            row["negative_lines"] = negative_lines
            got[row.get("type", "")] += 1
        else:
            why["no candidate kept"] += 1
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{sum(got.values())} of {len(rows)} rows have a negative ({', '.join(f'{k} {v}' for k, v in got.items())})")
    print("without: " + ", ".join(f"{k} {v}" for k, v in why.most_common()))
    print(f"-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("queries")
    a.add_argument("--pairs", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(act=queries)
    b = sub.add_parser("candidates")
    b.add_argument("--db", type=Path, required=True)
    b.add_argument("--pairs", type=Path, required=True)
    b.add_argument("--mined", type=Path, required=True)
    b.add_argument("--tag", required=True, help="the keys' prefix, one for each pairs file")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--k", type=int, default=K)
    b.add_argument("--hold", type=Path, action="append", default=[])
    b.add_argument("--hold-grades", type=Path, action="append", default=[])
    b.add_argument("--hold-lines", type=Path, action="append", default=[],
                   help="a kept-out test (-test.jsonl): no line of it, nor within three lines of one, is a candidate")
    b.set_defaults(act=candidates)
    c = sub.add_parser("apply")
    c.add_argument("--db", type=Path, required=True)
    c.add_argument("--pairs", type=Path, required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--spec", type=Path, required=True, help="what candidates wrote")
    c.add_argument("--scores", type=Path, required=True, help="the pool stage's reranked ranks file of that spec")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--below", type=float, default=BELOW)
    c.add_argument("--margin", type=float, default=MARGIN)
    c.add_argument("--per-row", type=int, default=PER_ROW)
    c.set_defaults(act=apply)
    args = ap.parse_args()
    args.act(args)


if __name__ == "__main__":
    main()

"""Training pairs for fine-tuning the model on the library, from the links seforim.db already has.

    python build_train_pairs.py --db seforim.db --out build/train/pairs.jsonl

Each link (commentary <-> its text, Ein Mishpat, Mesorat HaShas, Targum, quotation, Mishnah in the Talmud...) gives a
pair: a short query cut from one side (a span of SPAN_WORDS words, as a reader would type) and the other side as the
passage to find. Every type is capped (TYPE_CAPS) so the 2.3M commentary links do not drown the rest.

The benchmark of build_eval_set.py stays out: no link touching its books (the Shulchan Aruch, its five tractates), and
no Rambam halacha it queries with (those linked by the Ein Mishpat to the five tractates).

With --hold-evaluations every other evaluation stays out too, as in build_chavruta_pairs.py (RUN3_PLAN.md, section 4,
for teacher v2): no line an evaluation judges or targets, nor one linked to it or near it, nor one near a dev
question's target; no line of crfh1672/search-queries' tractates; no line matching eval-100's sources, holding a
quote's source, or copying a line of a book held out. It needs the evaluations' files (train-v2/, eval-*/): run it
here, and give its pairs to kaggle_run.py push train --train-pairs.
"""

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

from build_eval_set import SHULCHAN_ARUCH, TRACTATES
from pardes_semantic.clean import clean

# Per type, at most this many links (before the --pairs cap). OTHER and RELATED are loose: few of them.
TYPE_CAPS = {
    "COMMENTARY": 200_000, "SUPER_COMMENTARY": 20_000, "EIN_MISHPAT": 100_000, "MESORAT_HASHAS": 45_000,
    "TARGUM": 34_000, "QUOTATION": 31_000, "REFERENCE": 50_000, "MIDRASH": 7_000, "MISHNAH_IN_TALMUD": 2_400,
    "PARSHANUT": 2_000, "DIBUR_HAMATCHIL": 1_700, "RELATED": 20_000, "OTHER": 20_000,
}
SPAN_WORDS = (5, 20)
MIN_WORDS = 4
MAX_PASSAGE_WORDS = 220  # ~256 tokens: what the training reads of a passage
HERE = Path(__file__).resolve().parent


def benchmark_held(con: sqlite3.Connection) -> tuple[set[int], set[int]]:
    """What the benchmark measures and the training must not touch: its books (the Shulchan Aruch, its five
    tractates), and the lines the Ein Mishpat of those tractates links to (the Rambam halachot it queries with)."""
    titles = SHULCHAN_ARUCH + TRACTATES
    marks = ",".join("?" * len(titles))
    held_books = {r[0] for r in con.execute(f"SELECT id FROM book WHERE isBaseBook = 1 AND title IN ({marks})", titles)}
    tractates = [r[0] for r in con.execute(
        f"SELECT id FROM book WHERE isBaseBook = 1 AND title IN ({','.join('?' * len(TRACTATES))})", TRACTATES)]
    held_lines = {r[0] for r in con.execute(
        f"""SELECT k.targetLineId FROM link k JOIN connection_type ct ON ct.id = k.connectionTypeId
            WHERE ct.name = 'EIN_MISHPAT' AND k.sourceBookId IN ({','.join('?' * len(tractates))})""", tractates)}
    return held_books, held_lines


def evaluations_held(con: sqlite3.Connection, held_books: set[int]):
    """--hold-evaluations: whether a line (its id and its bookId, lineIndex, heRef, content) touches an evaluation, as
    build_chavruta_pairs.py tests both sides of a link; [held_books] gains crfh1672's tractates."""
    sys.path.insert(0, str(HERE / "train-v2"))
    from build_pairs import DEV_BATCHES, read_tsv
    from sample_targets import held_book_texts, held_out, neighbors, outside_set_books

    held_books |= outside_set_books(con)
    held_texts = held_book_texts(con, held_books)
    judged, patterns, quotes = held_out(con)
    linked, near = neighbors(con, judged)
    dev = {int(q["target_line"]) for q in read_tsv(HERE / "train-v2" / "questions.tsv") if q["batch"] in DEV_BATCHES}
    dev_linked, dev_near = neighbors(con, dev)
    plain_quotes = [q for q in quotes if q]
    marks = re.compile(r"[֑-ׇ]")

    def touches(line_id, row):
        book, index, ref, content = row
        plain = " ".join(re.sub(r"[^א-ת ]", " ", marks.sub("", content.replace("־", " "))).split())
        out = (book in held_books or line_id in linked or (book, index) in near or line_id in dev_linked
               or (book, index) in dev_near or (ref and any(p.search(ref) for p in patterns))
               or clean(content) in held_texts or any(q in plain for q in plain_quotes))
        touches.left_out += out
        return out

    touches.left_out = 0
    return touches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=5785)
    ap.add_argument("--hold-evaluations", action="store_true",
                    help="leave out what every evaluation measures too (build_chavruta_pairs.py's test)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    held_books, held_lines = benchmark_held(con)
    touches = evaluations_held(con, held_books) if args.hold_evaluations else None
    print(f"held out: {len(held_books)} books, {len(held_lines)} Rambam halachot")

    chosen: list[tuple[str, int, int]] = []
    for name, cap in TYPE_CAPS.items():
        rows = con.execute(
            """SELECT k.sourceLineId, k.targetLineId, k.sourceBookId, k.targetBookId FROM link k
               JOIN connection_type ct ON ct.id = k.connectionTypeId WHERE ct.name = ?""", (name,)).fetchall()
        rows = [r for r in rows if r[2] not in held_books and r[3] not in held_books
                and r[0] not in held_lines and r[1] not in held_lines]
        picked = rng.sample(rows, min(cap, len(rows)))
        chosen += [(name, a, b) for a, b, _, _ in picked]
        print(f"{name}: {len(picked)} of {len(rows)}")
    rng.shuffle(chosen)

    def text(line_id: int) -> str | None:
        row = con.execute("SELECT bookId, lineIndex, heRef, content FROM line WHERE id = ?", (line_id,)).fetchone()
        if row is None or (touches and touches(line_id, row)):
            return None
        return clean(row[3])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for name, a, b in chosen:
            if written >= args.pairs:
                break
            ta, tb = text(a), text(b)
            if not ta or not tb:
                continue
            # Either side may be the one the reader remembers
            if rng.random() < 0.5:
                ta, tb = tb, ta
            words, passage = ta.split(), tb.split()
            if len(words) < MIN_WORDS or len(passage) < MIN_WORDS or ta == tb:
                continue
            n = min(len(words), rng.randint(*SPAN_WORDS))
            start = rng.randint(0, len(words) - n)
            query = " ".join(words[start:start + n])
            f.write(json.dumps({"anchor": query, "positive": " ".join(passage[:MAX_PASSAGE_WORDS]), "type": name},
                               ensure_ascii=False) + "\n")
            written += 1
    print(f"{written} pairs -> {args.out}")
    if touches:
        print(f"{touches.left_out} lines left out as near what the evaluations measure")


if __name__ == "__main__":
    main()

"""Finds lines of the library to read them: by book title and reference, by words (niqqud and punctuation ignored), or
one line whole. For a judge who names a source from knowledge and must read it before grading it (INSTRUCTIONS.md).

    python judge/lookup.py --book "בבא בתרא" --words "שתי לבבות"
    python judge/lookup.py --ref "שולחן ערוך, אורח חיים קפח" --words שלישית
    python judge/lookup.py --line 80616

Prints the lines' text, for reading on the machine that holds the library: none of it goes into the repository.
"""

import argparse
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pardes_semantic.clean import clean  # noqa: E402


def plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", clean(text or ""))
    return re.sub(r"\s+", " ", "".join(c for c in text if not unicodedata.combining(c)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "build/library/seforim.db")
    ap.add_argument("--book", help="the book's title, or its beginning (base books first)")
    ap.add_argument("--ref", help="the beginning of the line's reference, e.g. 'שולחן ערוך, אורח חיים קפח'")
    ap.add_argument("--words", nargs="*", default=[], help="each must be in the line (a phrase in quotes stays together)")
    ap.add_argument("--line", type=int, help="print this line whole (its first 400 words)")
    ap.add_argument("--max", type=int, default=8)
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    if args.line:
        row = con.execute("SELECT l.content, l.heRef, b.title FROM line l JOIN book b ON b.id = l.bookId WHERE l.id = ?",
                          (args.line,)).fetchone()
        if not row:
            sys.exit("no such line")
        words = plain(row[0]).split()
        print(f"[{args.line}] {row[1] or row[2]} ({len(words)}w): {' '.join(words[:400])}")
        return
    if not args.book and not args.ref:
        sys.exit("give --book or --ref (the whole library cannot be scanned by words alone)")
    select = "SELECT id, title, isBaseBook FROM book WHERE "
    if args.book:
        order = " ORDER BY isBaseBook DESC, length(title) LIMIT 6"
        books = con.execute(select + "title = ? ORDER BY isBaseBook DESC", (args.book,)).fetchall() or \
            con.execute(select + "title LIKE ?" + order, (args.book + "%",)).fetchall() or \
            con.execute(select + "title LIKE ?" + order, ("%" + args.book + "%",)).fetchall()
        if not books:
            sys.exit(f"no book titled like {args.book}")
    else:
        # The reference begins with its book's title: the lines of that book only, not a scan of the whole library
        books = con.execute(select + "? LIKE title || '%' ORDER BY length(title) DESC LIMIT 3", (args.ref,)).fetchall()
        if not books:
            sys.exit(f"no book's title begins the reference {args.ref}: give --book")
    print("books:", "; ".join(f"{t}{'' if base else ' (not a base book)'}" for _, t, base in books))
    where, params = f"l.bookId IN ({','.join('?' * len(books))})", [b[0] for b in books]
    if args.ref:
        where += " AND l.heRef LIKE ?"
        params.append(args.ref + "%")
    needles = [plain(w) for w in args.words]
    found = 0
    for line_id, content, ref, title in con.execute(
            f"SELECT l.id, l.content, l.heRef, b.title FROM line l JOIN book b ON b.id = l.bookId WHERE {where} "
            "ORDER BY l.bookId, l.lineIndex", params):
        text = plain(content)
        if all(n in text for n in needles):
            found += 1
            if found <= args.max:
                words = text.split()
                first = next((k for k, w in enumerate(words) if needles and needles[0].split()[0] in w), 0)
                start = max(0, first - 12)
                print(f"[{line_id}] {ref or title} ({len(words)}w): {' '.join(words[start:start + 60])}")
    print(f"{found} lines" + (f", the first {args.max} shown" if found > args.max else ""))


if __name__ == "__main__":
    main()

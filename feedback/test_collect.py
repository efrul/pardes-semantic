"""collect.py's write_daf against packages whose tractate is not a plain name, which must not name a file elsewhere,
and what it makes of the searches and the grades of other apps that a package may hold:

    python feedback/test_collect.py      (or pytest)
"""

import csv
import io
import json
import sqlite3
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import ELSEWHERE, HISTORY, newest, place_results, read_packages, write_daf, write_history  # noqa: E402


def mark():
    return {"amud": "2a", "c": "gemara", "i": 3, "t": 0, "ok": True, "at": "2026-11-05T10:00:00.123456789Z"}


def test_only_a_plain_tractate_name_is_written():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "out"
        out.mkdir()
        daf = [
            ("a", {"tractate": "Berakhot", "marks": [mark()]}),
            ("b", {"tractate": "../../escaped", "marks": [mark()]}),
            ("c", {"tractate": str(root / "absolute"), "marks": [mark()]}),
        ]
        with redirect_stdout(io.StringIO()) as printed:
            write_daf(daf, out)
        assert sorted(p.name for p in (out / "daf").iterdir()) == ["Berakhot.json"]
        assert not (root / "escaped.json").exists()
        assert not (root / "absolute.json").exists()
        assert json.loads((out / "daf" / "Berakhot.json").read_text("utf-8"))["tractate"] == "Berakhot"
        assert "2 reviews left out" in printed.getvalue()


def test_what_is_wrong_with_a_line_goes_through_and_two_reasons_differ():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        wrong = dict(mark(), ok=False)
        daf = [
            ("a", {"tractate": "Berakhot", "marks": [dict(wrong, layout=True, to=4)]}),
            ("b", {"tractate": "Berakhot", "marks": [dict(wrong, words=True, at="2026-11-06T10:00:00Z")]}),
        ]
        with redirect_stdout(io.StringIO()) as printed:
            write_daf(daf, out)
        (kept,) = json.loads((out / "daf" / "Berakhot.json").read_text("utf-8"))["marks"]
        assert kept["words"] is True and "layout" not in kept, "the newest mark, as it was sent"
        assert "1 lines marked by more than one sender, 1 of them differently" in printed.getvalue()

        with redirect_stdout(io.StringIO()) as printed:
            write_daf([("a", {"tractate": "Shabbat", "marks": [{**wrong, "layout": True, "from": 1, "to": 4}]})], out)
        (kept,) = json.loads((out / "daf" / "Shabbat.json").read_text("utf-8"))["marks"]
        assert (kept["layout"], kept["from"], kept["to"]) == (True, 1, 4)


SENDER = "11111111-2222-3333-4444-555555555555"


def package(**rows):
    return {"format": 1, "app": "Pardes", "sender": SENDER, "sentAt": "2026-11-11T10:00:00Z", **rows}


def test_a_search_of_another_app_keeps_its_app_and_counts_apart():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        searches = [
            {"query": "q", "lastAt": 100, "count": 2},
            {"query": "q", "lastAt": 90, "count": 1, "source": "zayit"},
            {"query": "q", "lastAt": 80, "count": 1, "source": "otzaria", "scope": "s", "ref": "r", "timeUnknown": True},
            # The same search of Otzaria's in a later package: its newest stands
            {"query": "q", "lastAt": 85, "count": 1, "source": "otzaria", "scope": "s", "ref": "r", "timeUnknown": True},
        ]
        (root / "pardes-feedback-a.json").write_text(json.dumps(package(searches=searches)), "utf-8")
        _, history, _, senders = read_packages([root])
        assert senders == {SENDER: 1}
        kept = newest(history, time="lastAt",
                      key=lambda h: (h["sender"],) + tuple(h[k] for k in HISTORY[:5]) + tuple(h.get(k) for k in ELSEWHERE))
        assert [(h["source"], h["lastAt"]) for h in kept] == [("otzaria", 85), ("zayit", 90), (None, 100)]

        with redirect_stdout(io.StringIO()):
            write_history(kept, root)
        rows = list(csv.DictReader((root / "history.tsv").open(encoding="utf-8"), delimiter="\t"))
        assert [(r["source"], r["scope"], r["ref"]) for r in rows] == [("otzaria", "s", "r"), ("zayit", "", ""), ("", "", "")]
        assert rows[0]["sender"] == SENDER[:8]


def test_a_result_of_another_app_is_placed_by_its_book_and_reference():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE book (id INTEGER PRIMARY KEY, title TEXT)")
    con.execute("CREATE TABLE line (id INTEGER PRIMARY KEY, bookId INTEGER, heRef TEXT)")
    con.executemany("INSERT INTO book VALUES (?, ?)", [(7, "A"), (8, "B")])
    con.executemany("INSERT INTO line VALUES (?, ?, ?)", [(70, 7, "A 1"), (71, 7, "A 2"), (72, 7, "A 2"), (80, 8, "A 1")])

    def vote(title, ref):
        return {"lineId": 0, "bookId": 0, "bookTitle": title, "heRef": ref, "source": "otzaria"}

    own = {"lineId": 5, "bookId": 1}
    placed, left_out = place_results(con, [own, vote("A", "A 1"), vote("A", "A 2"), vote("A", "A 9"), vote(None, "A 1"),
                                           vote("C", "A 1")])
    # This library's own ids as they are; the one line the book and the reference name; and the rest left out: a
    # reference of two lines, of none, a result without its book, and a book this library lacks
    assert [(g["lineId"], g["bookId"]) for g in placed] == [(5, 1), (70, 7)]
    assert left_out == 4


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok {name}")

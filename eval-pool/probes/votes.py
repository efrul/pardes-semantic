"""A probe: what a user's marks on a page of results would add to the page after it (relevance feedback, the idea of
Otzaria's issue 5890: 👍 and 👎 on the results shown order the results not yet shown).

    python eval-pool/probes/votes.py <base.npz> [--know <know.npz>] [--feedback build/feedback/grades.tsv] [--one]

Run from the repository's root. The vectors are vectors.py's of one model: base over the pool's searches (crfh/, real/,
v1/, feedback/ and j/, the judged page's six lines and the sources the judges knew), know over the judged searches
with the model's 150 dense candidates (results-why.md).

The marks are the grades: a line graded 2 or 3 is marked relevant, 0 or 1 not relevant. The page after is the
candidates not yet shown, in the order of the query's vector moved toward the relevant lines and away from the others
(Rocchio), against the same candidates in the query's own order:

    q' = q + POS * mean(relevant) - NEG * mean(not relevant)

- real/, v1/, feedback/: the first page is the model's first three judged lines, the page after is the other judged
  lines. Only judged lines are ordered, as in ../judged_measure.py.
- crfh/: the first page is the model's first six lines of the tractate. Where the correct line is not on it, the six
  are marked not relevant (nothing else is known of them), and the question is whether the correct line reaches the
  next six.
- j/ with --know: the first page is the page the judges saw, marked by their grades. The page after is the model's 150
  candidates and the known sources, without the page's lines; a search counts when a good line (2 or 3) the page did
  not show is in the next six.

--one: a user who marks one line only, the page's first relevant line, or its first line as not relevant when it has
none ("more like this").
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["eval-pool"]
from judged_measure import grades_of  # noqa: E402

SETTINGS = [(0.75, 0.15), (0.75, 0.0), (0.0, 0.15), (1.0, 0.5), (0.5, 0.25)]  # (POS, NEG); the first is the textbook's
ONE = False


def marks(relevant: list, others: list):
    """The marks a user gives: all of them, or with --one a single line."""
    if not ONE:
        return relevant, others
    return (relevant[:1], []) if relevant else ([], others[:1])


def load(path):
    z = np.load(path, allow_pickle=True)
    docs = dict(zip(z["doc_ids"].tolist(), z["docs"]))
    queries = dict(zip([str(k) for k in z["keys"]], z["queries"]))
    return docs, queries


def moved(q, relevant, others, pos, neg):
    v = q.copy()
    if relevant and pos:
        v = v + pos * np.mean(relevant, axis=0)
    if others and neg:
        v = v - neg * np.mean(others, axis=0)
    return v / np.linalg.norm(v)


def ordered(lines, docs, q):
    return sorted(lines, key=lambda d: -float(docs[d] @ q))


def paired(rows, what):
    """rows: (before, after) booleans of each search."""
    b, a = sum(x for x, _ in rows), sum(y for _, y in rows)
    lost, won = sum(x and not y for x, y in rows), sum(y and not x for x, y in rows)
    bar = 2 * math.sqrt(lost + won)
    verdict = "within the noise" if abs(won - lost) <= bar else ("better" if won > lost else "worse")
    return f"{what}: {b} -> {a} of {len(rows)} (lost {lost}, won {won}, bar {bar:.1f}: {verdict})"


def judged_pool(name, graded, docs, queries, pos, neg, page=3):
    first, three, kinds = [], [], {"a relevant line marked": [], "only not relevant marked": []}
    for qid, lines in graded.items():
        q = queries.get(f"{name}/{qid}")
        cands = [d for d in lines if d in docs]
        if q is None or len(cands) < page + 2:
            continue
        shown = ordered(cands, docs, q)[:page]
        rest = [d for d in cands if d not in shown]
        if not any(lines[d] >= 2 for d in rest):
            continue  # nothing good is left to bring
        rel = [docs[d] for d in shown if lines[d] >= 2]
        oth = [docs[d] for d in shown if lines[d] < 2]
        rel, oth = marks(rel, oth)
        before = ordered(rest, docs, q)
        after = ordered(rest, docs, moved(q, rel, oth, pos, neg))
        row = (lines[before[0]] >= 2, lines[after[0]] >= 2)
        first.append(row)
        three.append((sum(lines[d] >= 2 for d in before[:3]), sum(lines[d] >= 2 for d in after[:3])))
        kinds["a relevant line marked" if rel else "only not relevant marked"].append(row)
    out = [paired(first, "the first line of the page after is good")]
    out.append(f"good lines in its first three: {sum(x for x, _ in three)} -> {sum(y for _, y in three)}")
    out += [paired(rows, f"  {kind}") for kind, rows in kinds.items() if rows]
    return out


def crfh(docs, queries, searches, pos, neg, page=6):
    rows, places = [], []
    for s in searches:
        q, correct = queries.get(s["key"]), s["correct"]
        cands = [d for d in s["candidates"] if d in docs]
        if q is None or correct not in docs:
            continue
        order = ordered(cands, docs, q)
        if correct in order[:page]:
            continue
        shown, rest = order[:page], order[page:]
        after = ordered(rest, docs, moved(q, *marks([], [docs[d] for d in shown]), pos, neg))
        rows.append((correct in rest[:page], correct in after[:page]))
        places.append((rest.index(correct) + 1, after.index(correct) + 1))
    med = lambda v: sorted(v)[len(v) // 2]
    return [paired(rows, "the correct line is in the next six"),
            f"its place among the lines not shown, median: {med([b for b, _ in places])} -> {med([a for _, a in places])}"]


def known(docs, queries, kdocs, kqueries, kcands, grades, pages, pos, neg, page=6):
    rows, kinds = [], {"a relevant line marked": [], "only not relevant marked": []}
    for key, cands in kcands.items():
        qid = key.split("/")[1]
        q, lines, shown = kqueries.get(key), grades.get(qid), pages.get(qid)
        if q is None or lines is None or not shown:
            continue
        vec = lambda d: kdocs.get(d, docs.get(d))
        rest = [d for d in dict.fromkeys(list(cands) + [d for d in lines if d not in shown]) if d not in shown and vec(d) is not None]
        if not any(lines.get(d, 0) >= 2 for d in rest):
            continue  # the judges knew no good line the page did not show
        rel = [vec(d) for d in shown if lines.get(d, 0) >= 2 and vec(d) is not None]
        oth = [vec(d) for d in shown if d in lines and lines[d] < 2 and vec(d) is not None]
        rel, oth = marks(rel, oth)
        both = {d: vec(d) for d in rest}
        before = ordered(rest, both, q)
        after = ordered(rest, both, moved(q, rel, oth, pos, neg))
        row = (any(lines.get(d, 0) >= 2 for d in before[:page]), any(lines.get(d, 0) >= 2 for d in after[:page]))
        rows.append(row)
        kinds["a relevant line marked" if rel else "only not relevant marked"].append(row)
    return [paired(rows, "a good line the page did not show is in the next six")] + [
        paired(r, f"  {kind}") for kind, r in kinds.items() if r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("--know", type=Path)
    ap.add_argument("--spec", type=Path, default=Path("build/rerank/pool-base.json"))
    ap.add_argument("--know-spec", type=Path, default=Path("build/why/spec-know.json"))
    ap.add_argument("--feedback", type=Path)
    ap.add_argument("--one", action="store_true", help="one mark a page, not a mark on every line")
    args = ap.parse_args()
    global ONE
    ONE = args.one
    docs, queries = load(args.base)
    spec = json.loads(args.spec.read_text("utf-8"))["searches"]
    right = {r["id"]: int(r["correct_line"]) for r in csv.DictReader(open("eval-crfh/lines.tsv", encoding="utf-8"), delimiter="\t")}
    crfh_searches = [{"key": s["key"], "candidates": s["candidates"], "correct": right[s["key"].split("/")[1]]}
                     for s in spec if s["key"].startswith("crfh/") and s["key"].split("/")[1] in right]
    pools = {"real": grades_of(Path("eval-real/grades.tsv")), "v1": grades_of(Path("eval-v1/grades.tsv"))}
    if args.feedback:
        pools["feedback"] = grades_of(args.feedback)
    judged = grades_of(Path("judge/grades.tsv"))
    for pos, neg in SETTINGS:
        print(f"== POS {pos}, NEG {neg}")
        for name, graded in pools.items():
            for line in judged_pool(name, graded, docs, queries, pos, neg):
                print(f"{name}: {line}")
        if neg:
            for line in crfh(docs, queries, crfh_searches, pos, neg):
                print(f"crfh: {line}")
        if args.know:
            kdocs, kqueries = load(args.know)
            kspec = json.loads(args.know_spec.read_text("utf-8"))["searches"]
            key = {r["qid"]: r["query"] for r in csv.DictReader(open("build/judge-pick/key.tsv", encoding="utf-8"),
                                                               delimiter="\t", quoting=csv.QUOTE_NONE)}
            runs = {}
            for p in ("build/judge/run.jsonl", "build/demo/run.jsonl"):
                for row in open(p, encoding="utf-8"):
                    r = json.loads(row)
                    runs.setdefault(r["query"], [x["lineId"] for x in r["hybrid"][:6]])
            pages = {qid: runs[query] for qid, query in key.items() if query in runs}
            for line in known(docs, queries, kdocs, kqueries, {s["key"]: s["candidates"] for s in kspec}, judged, pages, pos, neg):
                print(f"judged: {line}")


if __name__ == "__main__":
    main()

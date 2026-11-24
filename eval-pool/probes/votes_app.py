"""A probe: marks on a search's first lines, in the app's own engines over the whole library (votes.py simulates it on
one model's vectors of the judged lines; this is search-order/harness/Votes.kt, the fused list as the app builds it and
the dense side searched again with the moved vector).

    python eval-pool/probes/votes_app.py spec --db seforim.db --out build/votes/searches.jsonl [--feedback grades.tsv]
    java -cp "$(cat build/harness/classpath.txt)" VotesKt seforim.db build/votes/searches.jsonl build/votes/run.jsonl 6 0.75 0.15
    python eval-pool/probes/votes_app.py measure build/votes/searches.jsonl build/votes/run.jsonl [--first 6] [--next 6]

Run from the repository's root. spec writes the searches of pool.json's (--spec) with what is known of their lines:
real/, v1/, feedback/ and j/ with their grades, crfh/ with its correct line, in its tractate alone, every other line of
its first lines counted as not relevant (nothing else is known of them). The queries stay under build/.

measure: a search counts when it was marked and a good line (2 or 3) is known that its first lines did not show; the
question is whether one is among the next lines, both orders without the first lines (they stay where the reader saw
them): the list as it was against the list searched again with the marks, the dense side alone with and without them,
and the list against the dense side with them. A line no one graded counts as not good, and the grades were given to
the first lines of earlier lists: "the graded lines alone" leaves the others out of both orders.
"""
import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path[:0] = ["eval-pool"]
from judged_measure import grades_of  # noqa: E402


def spec(args) -> None:
    searches = json.loads(args.spec.read_text("utf-8"))["searches"]
    pools = {"real": grades_of(Path("eval-real/grades.tsv")), "v1": grades_of(Path("eval-v1/grades.tsv")),
             "j": grades_of(Path("judge/grades.tsv"))}
    if args.feedback:
        pools["feedback"] = grades_of(args.feedback)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    right = {r["id"]: int(r["correct_line"]) for r in csv.DictReader(open("eval-crfh/lines.tsv", encoding="utf-8"), delimiter="\t")}
    n = defaultdict(int)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for s in searches:
            name, qid = s["key"].split("/", 1)
            row = {"key": s["key"], "query": s["query"]}
            if name == "crfh":
                if qid not in right:
                    continue
                row["grades"] = {str(right[qid]): 3}
                row["book"] = con.execute("SELECT bookId FROM line WHERE id = ?", (right[qid],)).fetchone()[0]
                row["unknown"] = 0
            elif qid in pools.get(name, {}):
                row["grades"] = {str(line): grade for line, grade in pools[name][qid].items()}
            else:
                continue
            n[name] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(", ".join(f"{k} {v}" for k, v in n.items()), "->", args.out)


def paired(rows, what):
    b, a = sum(x for x, _ in rows), sum(y for _, y in rows)
    lost, won = sum(x and not y for x, y in rows), sum(y and not x for x, y in rows)
    bar = 2 * math.sqrt(lost + won)
    verdict = "within the noise" if abs(won - lost) <= bar else ("better" if won > lost else "worse")
    return f"{what}: {b} -> {a} of {len(rows)} (lost {lost}, won {won}, bar {bar:.1f}: {verdict})"


ARMS = [("list", "moved", "the list, searched again with the marks"),
        ("dense", "denseMoved", "the dense side alone, with the marks"),
        ("list", "denseMoved", "the list against the dense side with the marks")]


def measure(args) -> None:
    known = {}
    for raw in args.searches.open(encoding="utf-8"):
        s = json.loads(raw)
        known[s["key"]] = {int(k): v for k, v in s["grades"].items()}
    rows = defaultdict(lambda: defaultdict(list))   # (set, kind, arms) -> measure -> [(before, after)]
    seen, marked, graded, times = defaultdict(int), defaultdict(int), defaultdict(lambda: defaultdict(float)), []
    for raw in args.run.open(encoding="utf-8"):
        r = json.loads(raw)
        name, grades = r["key"].split("/")[0], known[r["key"]]
        seen[name] += 1
        if "moved" not in r:
            continue
        marked[name] += 1
        times.append(r["movedMs"])
        page = r["list"][:args.first]
        good = lambda d: grades.get(d, 0) >= 2
        if not any(good(d) for d in grades if d not in page):
            continue  # no good line is known that the first lines did not show
        after = {k: [d for d in r[k] if d not in page] for k in ("list", "moved", "dense", "denseMoved") if k in r}
        for k, lines in after.items():
            graded[name][k] += sum(d in grades for d in lines[:args.next])
        graded[name]["searches"] += 1
        for a, b, what in ARMS:
            if a not in after or b not in after:
                continue
            ga, gb = [d for d in after[a] if d in grades], [d for d in after[b] if d in grades]
            for kind in ("every marked search", "a relevant line marked" if r["relevant"] else "only not relevant marked"):
                at = rows[(name, kind, what)]
                at[f"a good line in the next {args.next}"].append((any(map(good, after[a][:args.next])), any(map(good, after[b][:args.next]))))
                at["in the next 25"].append((any(map(good, after[a][:25])), any(map(good, after[b][:25]))))
                at["the graded lines alone: the first is good"].append((bool(ga) and good(ga[0]), bool(gb) and good(gb[0])))
    for name in seen:
        print(f"== {name}: {seen[name]} searches, {marked[name]} with a mark among their first {args.first}")
        for (set_, kind, what), measures in rows.items():
            if set_ == name:
                print(f"{what}, {kind}:")
                for measure_, pairs in measures.items():
                    print("  " + paired(pairs, measure_))
        g = graded[name]
        if g["searches"]:
            print(f"graded lines among the next {args.next}, a search: " + ", ".join(
                f"{k} {g[k] / g['searches']:.1f}" for k in ("list", "moved", "dense", "denseMoved") if k in g))
    if times:
        times.sort()
        print(f"the search with the marks: median {times[len(times) // 2]} ms, 90% under {times[len(times) * 9 // 10]} ms (a reader opened each time)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("spec")
    a.add_argument("--db", type=Path, required=True)
    a.add_argument("--spec", type=Path, default=Path("build/rerank/pool-base.json"))
    a.add_argument("--feedback", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(act=spec)
    b = sub.add_parser("measure")
    b.add_argument("searches", type=Path)
    b.add_argument("run", type=Path)
    b.add_argument("--first", type=int, default=6)
    b.add_argument("--next", type=int, default=6)
    b.set_defaults(act=measure)
    args = ap.parse_args()
    args.act(args)


if __name__ == "__main__":
    main()

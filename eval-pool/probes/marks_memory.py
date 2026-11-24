"""A probe: a memory of each person's own marks, as the app would use it. Of the searches users sent and graded
(feedback/collect.py's history.tsv, queries.tsv and grades.tsv, outside the repository), in the order of their times:

- how many of a person's searches came after that person had graded the same search or a near one (cosine NEAR of the
  pack's model): where the memory would have acted, with the marks people gave as things were;
- what the memory's rule does there: the lines the person marked relevant before go first, the lines marked not
  relevant go down. Checked against the grades the person gave this time.

    python eval-pool/probes/marks_memory.py build/feedback [--near 0.85] [--dump build/votes/memory.jsonl]
        [--placed build/votes/memory-run.jsonl] [--queries queries.txt] [--vectors queries.vec | --compare queries.vec]

--queries, --vectors and --compare check the threshold on the app's own vectors of the queries
(search-order/harness/QueryVectors.kt); --dump and --placed, where the remembered lines stand in the app's list of the
later search (search-order/harness/Votes.kt).

Two senders that share half or more of the smaller one's searches are one person's two installations (marks_reach.py).
Prints numbers only.
"""
import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["."]
import pool_rank  # noqa: E402
from pardes_semantic import model as m  # noqa: E402

MARKS = re.compile("[֑-ׇ]")
AGAIN = 10 * 60 * 1000   # a search run this long after its grades is another visit, not the one they were given in
FIRST = 6


def norm(s):
    return " ".join(re.sub(r"[\"'׳״.,:;!?()\[\]\-–—<>/]", " ", MARKS.sub("", s or "").replace("־", " ")).split())


def read(path):
    return list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))


def people_of(history):
    """{sender: person}: senders whose searches are mostly another's are merged."""
    sets = collections.defaultdict(set)
    for r in history:
        if norm(r["query"]):
            sets[r["sender"]].add(norm(r["query"]))
    parent = {s: s for s in sets}

    def find(s):
        parent.setdefault(s, s)
        while parent[s] != s:
            s = parent[s]
        return s
    order = sorted(sets, key=lambda s: -len(sets[s]))
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            small = min(len(sets[a]), len(sets[b]))
            if small >= 5 and len(sets[a] & sets[b]) / small >= 0.5:
                parent[find(b)] = find(a)
    return find


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--near", type=float, default=0.85)
    ap.add_argument("--model", default="build/kaggle-output/tune3/tuned")
    ap.add_argument("--queries", help="write the queries embedded, a line each, for search-order/harness/QueryVectors.kt")
    ap.add_argument("--vectors", help="QueryVectors.kt's vectors of --queries' file: the app's own, in place of the model's")
    ap.add_argument("--compare", help="QueryVectors.kt's vectors, compared with the model's")
    ap.add_argument("--placed", help="Votes.kt's output on --dump's file: where the remembered lines stand in its lists")
    ap.add_argument("--dump", help="write the searches the memory acts on, with the lines remembered, as "
                                   "search-order/harness/Votes.kt reads them: where those lines stand in the app's list")
    args = ap.parse_args()
    torch.set_num_threads(2)
    history = read(f"{args.dir}/history.tsv")
    person = people_of(history)
    grades = collections.defaultdict(dict)
    for g in read(f"{args.dir}/grades.tsv"):
        grades[g["qid"]][int(g["lineId"])] = (int(g["grade"]), int(g["rank"]))
    graded = [{"qid": q["qid"], "query": norm(q["query"]), "person": person(q["sender"]), "at": int(q["gradedAt"]),
               "lines": grades[q["qid"]]}
              for q in read(f"{args.dir}/queries.tsv") if q["qid"] in grades and norm(q["query"])]
    graded.sort(key=lambda s: s["at"])
    # A person's searches in Pardes, each once, at the time of its last run
    runs = {}
    for r in history:
        q = norm(r["query"])
        if not q or r["source"] or not (r["lastAt"] or "").isdigit():
            continue
        key = (person(r["sender"]), q)
        runs[key] = max(runs.get(key, 0), int(r["lastAt"]))
    texts = sorted({s["query"] for s in graded} | {q for _, q in runs})
    if args.queries:
        Path(args.queries).write_text("\n".join(texts) + "\n", "utf-8")
    if args.vectors:   # the app's own vectors of the same queries
        rows = np.loadtxt(args.vectors, dtype=np.float32)
        assert len(rows) == len(texts), (len(rows), len(texts))
        vectors = dict(zip(texts, rows / np.linalg.norm(rows, axis=1, keepdims=True)))
    else:
        loaded = m.load("cpu", torch.float32, args.model)
        vectors = dict(zip(texts, pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + t for t in texts])))
    if args.compare:   # how far the app's vectors are from these, by the pairs of queries the threshold tells apart
        rows = np.loadtxt(args.compare, dtype=np.float32)
        theirs = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        ours = np.stack([vectors[t] for t in texts])
        a, b = ours @ ours.T, theirs @ theirs.T
        upper = np.triu_indices(len(texts), 1)
        a, b = a[upper], b[upper]
        print(f"pairs of queries at {args.near}+: {int((a >= args.near).sum())} here, {int((b >= args.near).sum())} by the "
              f"app's vectors, {int(((a >= args.near) & (b >= args.near)).sum())} by both; the two cosines differ by "
              f"{float(np.abs(a - b).mean()):.3f} on average, {float(np.abs(a - b)[a >= 0.7].max()):.3f} at most among pairs at 0.7+")
    by_person = collections.defaultdict(list)
    for s in graded:
        by_person[s["person"]].append(s)
    print(f"{len(graded)} graded searches of {len(by_person)} people; {len(runs)} searches in Pardes of "
          f"{len({p for p, _ in runs})} people, {sum(1 for p, _ in runs if p in by_person)} of them by people who graded")

    def earlier(p, query, at, same_gap=0):
        """The person's graded searches before [at]: the same words, and near ones."""
        same, near = [], []
        for s in by_person.get(p, []):
            if s["query"] == query:
                if s["at"] < at - same_gap:
                    same.append(s)
            elif s["at"] < at and float(vectors[s["query"]] @ vectors[query]) >= args.near:
                near.append(s)
        return same, near

    # Where the memory would have acted, as things were
    acted = collections.Counter()
    for (p, query), at in runs.items():
        same, near = earlier(p, query, at, AGAIN)
        acted["the same search, graded on an earlier visit"] += bool(same)
        acted["a near search graded before"] += bool(near)
        acted["either"] += bool(same or near)
    for what, n in acted.items():
        print(f"the memory would have acted: {what}: {n} of {len(runs)} ({100 * n / len(runs):.1f}%)")
    mine = [k for k in runs if k[0] in by_person]
    n = sum(1 for (p, query) in mine if any(earlier(p, query, runs[(p, query)], AGAIN)))
    print(f"among the people who graded: {n} of {len(mine)} ({100 * n / max(1, len(mine)):.1f}%)")

    # The rule, on the searches a person graded after grading a near one
    with_memory = seen = good_first = was_first = 0
    dump = []
    liked_seen = liked_good = liked_unseen = 0
    bad_seen = bad_still = bad_on_first = 0
    places = []
    for s in graded:
        _, near = earlier(s["person"], s["query"], s["at"])
        if not near:
            continue
        with_memory += 1
        remembered = {}
        for other in near:                       # the later grade of a line wins
            for line, (grade, _) in other["lines"].items():
                remembered[line] = grade
        here = s["lines"]
        liked = sorted((line for line, g in remembered.items() if g >= 2), key=lambda line: -remembered[line])
        dump.append({"key": s["qid"], "query": s["query"], "grades": {str(line): g for line, (g, _) in here.items()},
                     "liked": liked, "disliked": [line for line, g in remembered.items() if g == 0]})
        for line in liked:
            if line in here:
                liked_seen += 1
                liked_good += here[line][0] >= 2
            else:
                liked_unseen += 1
        for line, g in remembered.items():
            if g == 0 and line in here:
                bad_seen += 1
                bad_still += here[line][0] == 0
                bad_on_first += here[line][0] == 0 and here[line][1] < FIRST
        graded_here = [line for line in liked if line in here]
        if graded_here:
            seen += 1
            good_first += here[graded_here[0]][0] >= 2
            first = [line for line, (g, rank) in here.items() if rank == 0]
            was_first += bool(first) and here[first[0]][0] >= 2
            good_places = [rank for g, rank in here.values() if g >= 2]
            if good_places:
                places.append(min(good_places))
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for row in dump:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.placed:   # where the remembered lines stand in the app's list of the later search (Votes.kt on --dump's file)
        by_key = {row["key"]: row for row in dump}
        liked_in, liked_out, liked_first, bad_in, bad_first, bad_all = [], 0, 0, 0, 0, 0
        searches = collections.Counter()
        for raw in open(args.placed, encoding="utf-8"):
            r = json.loads(raw)
            row, place = by_key[r["key"]], {line: i + 1 for i, line in enumerate(r["list"])}
            got = [place[line] for line in row["liked"] if line in place]
            searches["with a remembered relevant line"] += bool(row["liked"])
            searches["one of them in the list"] += bool(got)
            searches[f"one of them among its first {FIRST}"] += any(x <= FIRST for x in got)
            liked_in += got
            liked_out += len(row["liked"]) - len(got)
            bad_all += len(row["disliked"])
            bad_in += sum(line in place for line in row["disliked"])
            bad_first += sum(place.get(line, 10 ** 9) <= FIRST for line in row["disliked"])
        liked_in.sort()
        print("in the app's list today: " + ", ".join(f"{k} {v}" for k, v in searches.items()))
        print(f"remembered relevant lines: {len(liked_in)} in the list (median place {liked_in[len(liked_in) // 2] if liked_in else '-'}, "
              f"{sum(x <= FIRST for x in liked_in)} among its first {FIRST}), {liked_out} not in it")
        print(f"remembered not relevant lines: {bad_all}, {bad_in} in the list, {bad_first} among its first {FIRST}")
    print(f"{with_memory} graded searches came after a near graded search of the same person")
    print(f"lines marked relevant before: {liked_seen} graded again here, {liked_good} relevant here too; "
          f"{liked_unseen} not graded here")
    print(f"lines marked not relevant before: {bad_seen} graded again here, {bad_still} not relevant here too, "
          f"{bad_on_first} of those among this search's first {FIRST}")
    print(f"searches where a remembered relevant line was graded again: {seen}; the first place is good: "
          f"{was_first} as it was -> {good_first} with the remembered line first")
    if places:
        places.sort()
        print(f"the place of the first line the person marked good there, as it was: median {places[len(places) // 2] + 1}, "
              f"first place in {sum(p == 0 for p in places)}, among the first {FIRST} in {sum(p < FIRST for p in places)} of {len(places)}")


if __name__ == "__main__":
    main()

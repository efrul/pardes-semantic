"""A probe: do marks carry to the next search of the same need? Of the searches users graded (feedback/collect.py's
queries.tsv and grades.tsv, outside the repository): where a graded search is near another graded search, of the same
person or of another, are the lines graded in both graded alike?

    python eval-pool/probes/marks_transfer.py build/feedback

Prints numbers only.
"""
import collections, csv, re, sys
import numpy as np, torch
sys.path[:0] = ["."]
import pool_rank
from pardes_semantic import model as m
torch.set_num_threads(3)
D = sys.argv[1]
MARKS = re.compile("[֑-ׇ]")
def norm(s):
    return " ".join(re.sub(r"[\"'׳״.,:;!?()\[\]\-–—<>/]", " ", MARKS.sub("", s or "").replace("־", " ")).split())
hist = list(csv.DictReader(open(f"{D}/history.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
qsets = collections.defaultdict(set)
for r in hist:
    if norm(r["query"]): qsets[r["sender"]].add(norm(r["query"]))
parent = {s: s for s in qsets}
def find(s):
    parent.setdefault(s, s)
    while parent[s] != s: s = parent[s]
    return s
ss = sorted(qsets, key=lambda s: -len(qsets[s]))
for i, a in enumerate(ss):
    for b in ss[i + 1:]:
        small = min(len(qsets[a]), len(qsets[b]))
        if small >= 5 and len(qsets[a] & qsets[b]) / small >= 0.5: parent[find(b)] = find(a)
queries = list(csv.DictReader(open(f"{D}/queries.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
grades = collections.defaultdict(dict)
for g in csv.DictReader(open(f"{D}/grades.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
    grades[g["qid"]][int(g["lineId"])] = int(g["grade"])
items = [(q["qid"], norm(q["query"]), find(q["sender"])) for q in queries if q["qid"] in grades and norm(q["query"])]
print(len(items), "graded searches of", len({p for _, _, p in items}), "people;", sum(len(grades[i[0]]) for i in items), "grades")
loaded = m.load("cpu", torch.float32, "build/kaggle-output/tune3/tuned")
V = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + q for _, q, _ in items])
S = V @ V.T
np.fill_diagonal(S, -1)
for label, same in (("the same person's other graded searches", True), ("other people's graded searches", False)):
    for th in (0.85, 0.7):
        has = shared_lines = agree = with_shared = liked_there_liked_here = liked_there_seen_here = 0
        for k, (qid, q, p) in enumerate(items):
            near = [j for j in range(len(items)) if j != k and S[k, j] >= th and ((items[j][2] == p) == same) and items[j][1] != q]
            if not near: continue
            has += 1
            mine = grades[qid]
            got = False
            for j in near:
                theirs = grades[items[j][0]]
                for line, g in theirs.items():
                    if line in mine:
                        shared_lines += 1; got = True
                        agree += (g >= 2) == (mine[line] >= 2)
                        if g >= 2:
                            liked_there_seen_here += 1
                            liked_there_liked_here += mine[line] >= 2
            with_shared += got
        print(f"{label}, cosine {th}+: {has} searches have one | a line graded in both in {with_shared} | such lines {shared_lines}, graded alike {agree}"
              f" | liked there and graded here: {liked_there_seen_here}, liked here too: {liked_there_liked_here}")

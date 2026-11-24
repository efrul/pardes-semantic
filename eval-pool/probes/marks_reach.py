"""A probe: how far a memory of marks would reach. Of the searches users sent (feedback/collect.py's history.tsv, outside
the repository): how many come again to the person who made them, word for word or in other words (cosine 0.85 of the
pack's model), and how many another person made.

    python eval-pool/probes/marks_reach.py build/feedback

Two senders that share half or more of the smaller one's searches are one person's two installations, and are merged
first. Prints numbers only.
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
rows = list(csv.DictReader(open(f"{D}/history.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
runs = collections.defaultdict(collections.Counter)
for r in rows:
    q = norm(r["query"])
    if q: runs[r["sender"]][q] += int(r["count"]) if (r.get("count") or "").isdigit() else 1
senders = sorted(runs, key=lambda s: -len(runs[s]))
# a sender most of whose searches are another's is the same person on another installation
parent = {s: s for s in senders}
def find(s):
    while parent[s] != s: s = parent[s]
    return s
pairs = []
for i, a in enumerate(senders):
    for b in senders[i + 1:]:
        shared = len(set(runs[a]) & set(runs[b]))
        small = min(len(runs[a]), len(runs[b]))
        if small >= 5 and shared / small >= 0.5:
            pairs.append((len(runs[a]), len(runs[b]), shared))
            parent[find(b)] = find(a)
people = collections.defaultdict(collections.Counter)
for s in senders:
    for q, c in runs[s].items():
        people[find(s)][q] = max(people[find(s)][q], c)
print("pairs of senders that share half or more of the smaller one's searches:", pairs)
ps = sorted(people, key=lambda p: -len(people[p]))
sizes = [len(people[p]) for p in ps]
print(f"{len(senders)} senders -> {len(ps)} people | searches a person: the five largest {sizes[:5]} | median {sorted(sizes)[len(sizes)//2]} | with 20 or more: {sum(x >= 20 for x in sizes)}")
qs = sorted({q for c in people.values() for q in c})
loaded = m.load("cpu", torch.float32, "build/kaggle-output/tune3/tuned")
V = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + q for q in qs])
ix = {q: k for k, q in enumerate(qs)}
who = collections.defaultdict(set)
for p, c in people.items():
    for q in c: who[q].add(p)
def reach(group, label):
    n = again = own_near = other_same = other_near = any_own = any_other = 0
    for p in group:
        mine = np.array([ix[q] for q in people[p]])
        others = np.array([ix[q] for q in qs if who[q] - {p}])
        Sm = V[mine] @ V[mine].T
        np.fill_diagonal(Sm, -1)
        own_best = Sm.max(1) if len(mine) > 1 else np.full(len(mine), -1.0)
        oth_best = (V[mine] @ V[others].T).max(1) if len(others) else np.full(len(mine), -1.0)
        for k, q in enumerate(people[p]):
            n += 1
            a = people[p][q] > 1; b = own_best[k] >= 0.85; c = len(who[q]) > 1; d = oth_best[k] >= 0.85
            again += a; own_near += b; other_same += c; other_near += d; any_own += a or b; any_other += c or d
    pc = lambda x: f"{x} ({100 * x / max(1, n):.0f}%)"
    print(f"{label}: {n} searches | run again by the same person {pc(again)} | near another of that person's {pc(own_near)} | either {pc(any_own)}"
          f" || another person made it, word for word {pc(other_same)} | near another person's {pc(other_near)}")
reach(ps, "all people")
reach(ps[1:], "without the largest")
shared = sorted(((len(v), q) for q, v in who.items() if len(v) >= 3), reverse=True)
print("searches three people or more made:", len(shared), "| by words:", collections.Counter(min(len(q.split()), 4) for _, q in shared).most_common())

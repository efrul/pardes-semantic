"""The judged searches under several rerankers: the page's six in each reranker's order, against the page as it is.

    python eval-pool/probes/page_rerank.py name=ranks.json [name=ranks.json ...] [--fuse]

Run from the repository's root. Each ranks file is pool_rerank.py's over the pool's j/ searches (a search's judged
lines: the page's six of pack 4 and the sources the judges knew). Prints, of the searches with a judged page:
the first line good over the six alone and with the known sources among the candidates, the paired count against the
page and against the first reranker named, and the same by how many of the query's words the good line shares.
"""
import csv
import json
import math
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
FUSE = "--fuse" in sys.argv
models = {}
for a in args:
    name, _, path = a.partition("=")
    models[name] = json.load(open(path))
status = {r["qid"]: r["status"] for r in csv.DictReader(open("judge/status.tsv", encoding="utf-8"), delimiter="\t")}
key = {r["qid"]: r["query"] for r in csv.DictReader(open("build/judge-pick/key.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)}
grades = {}
for r in csv.DictReader(open("judge/grades.tsv", encoding="utf-8"), delimiter="\t"):
    grades.setdefault(r["qid"], {})[int(r["lineId"])] = int(r["grade"])
runs = {}
for p in ("build/judge/run.jsonl", "build/demo/run.jsonl"):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        runs.setdefault(r["query"], r)
overlap = json.load(open("build/why/overlap.json"))
ok = [q for q in grades if status.get(q, "ok") == "ok" and key[q] in runs and all(f"j/{q}" in m for m in models.values())]


def page(q):
    return [x["lineId"] for x in runs[key[q]]["hybrid"][:6] if x["lineId"] in grades[q]]


def order(name, q, page_only):
    p = set(page(q))
    return [d for d in models[name][f"j/{q}"]["ids"] if d in grades[q] and (not page_only or d in p)]


def fused(name, q, w=4.0, k=3):
    p, r = page(q), order(name, q, True)
    return sorted(p, key=lambda d: -(1 / (k + p.index(d)) + (w / (k + r.index(d)) if d in r else 0)))


def good(o, q, n=1):
    return any(grades[q][d] >= 2 for d in o[:n])


def source(o, q):
    return bool(o) and grades[q][o[0]] == 3


def paired(a, b):
    x = sum(a[q] and not b[q] for q in ok)
    y = sum(b[q] and not a[q] for q in ok)
    return x, y, 2 * math.sqrt(x + y)


base = {q: good(page(q), q) for q in ok}
print(f"{len(ok)} judged searches; the page as it is: first good {sum(base.values())}, good in three "
      f"{sum(good(page(q), q, 3) for q in ok)}, the source first {sum(source(page(q), q) for q in ok)}")
first = next(iter(models))
six = {}
for name in models:
    six[name] = {q: good(order(name, q, True), q) for q in ok}
    full = {q: good(order(name, q, False), q) for q in ok}
    a, b, bar = paired(base, six[name])
    line = (f"{name:28s} the six: first good {sum(six[name].values())} (only the page {a}, only it {b}, bar {bar:.1f}), "
            f"the source first {sum(source(order(name, q, True), q) for q in ok)} | with the sources: first good "
            f"{sum(full.values())}, in three {sum(good(order(name, q, False), q, 3) for q in ok)}, the source first "
            f"{sum(source(order(name, q, False), q) for q in ok)}")
    if name != first:
        a, b, bar = paired(six[first], six[name])
        line += f" | against {first}: only {first} {a}, only it {b}, bar {bar:.1f}"
    print(line)
    if FUSE:
        for w in (1.0, 2.0):
            f = {q: good(fused(name, q, w), q) for q in ok}
            a, b, bar = paired(base, f)
            print(f"{'':28s} fused with the page's order, weight {w}: {sum(f.values())} (only the page {a}, only fused {b}, bar {bar:.1f})")
print("by the words the good line shares with the query (first good over the six):")
classes = sorted(set(overlap.values()))
for c in classes:
    qs = [q for q in ok if overlap.get(q) == c]
    print(f"   {c:16s} {len(qs):3d}: the page {sum(base[q] for q in qs)}; " + "; ".join(f"{n} {sum(six[n][q] for q in qs)}" for n in models))
qs = [q for q in ok if q not in overlap]
print(f"   {'no good line':16s} {len(qs):3d}")

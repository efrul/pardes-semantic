"""A probe: models paired on the judged searches, each ordering a search's judged lines (the page's six of pack 4 and
the sources the judges knew): the first line good (graded 2 or 3), and the first line a source (3).

    python eval-pool/probes/judged_pairs.py name=ranks.json name=ranks.json ... [--categories]

Run from the repository's root. Every pair of models is compared search by search, against twice the noise
(2 * sqrt(b + c)). --categories: of the sources each model puts first, the library's top category.
The candidates were chosen by pack 4's page: a source a judge added is a line the page missed, so the pack's model,
and its family less so, start behind any other model here (results-google.md).
"""
import collections
import csv
import json
import math
import sqlite3
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
models = {}
for a in args:
    name, _, path = a.partition("=")
    models[name] = json.load(open(path, encoding="utf-8"))
status = {r["qid"]: r["status"] for r in csv.DictReader(open("judge/status.tsv", encoding="utf-8"), delimiter="\t")}
key = {r["qid"]: r["query"] for r in csv.DictReader(open("build/judge-pick/key.tsv", encoding="utf-8"), delimiter="\t",
                                                    quoting=csv.QUOTE_NONE)}
grades = {}
for r in csv.DictReader(open("judge/grades.tsv", encoding="utf-8"), delimiter="\t"):
    grades.setdefault(r["qid"], {})[int(r["lineId"])] = int(r["grade"])
runs = set()
for p in ("build/judge/run.jsonl", "build/demo/run.jsonl"):
    for line in open(p, encoding="utf-8"):
        runs.add(json.loads(line)["query"])
ok = [q for q in grades if status.get(q, "ok") == "ok" and key[q] in runs and all(f"j/{q}" in m for m in models.values())]


def first(name, q):
    """The first of the search's judged lines in the model's order."""
    return next((d for d in models[name][f"j/{q}"]["ids"] if d in grades[q]), None)


print(f"{len(ok)} judged searches, {sum(3 in grades[q].values() for q in ok)} with a source judged")
names = list(models)
for least, what in ((2, "first good"), (3, "the source first")):
    got = {n: {q: first(n, q) is not None and grades[q][first(n, q)] >= least for q in ok} for n in names}
    print(f"{what}: " + ", ".join(f"{n} {sum(got[n].values())}" for n in names))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x = sum(got[a][q] and not got[b][q] for q in ok)
            y = sum(got[b][q] and not got[a][q] for q in ok)
            bar = 2 * math.sqrt(x + y)
            verdict = "within the noise" if abs(x - y) <= bar else f"{a if x > y else b} better"
            print(f"   {a} / {b}: only {a} {x}, only {b} {y}, bar {bar:.1f}: {verdict}")
if "--categories" in sys.argv:
    con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
    parent = {i: (p, t) for i, p, t in con.execute("SELECT id, parentId, title FROM category")}

    def root(line):
        c = con.execute("SELECT b.categoryId FROM line l JOIN book b ON b.id = l.bookId WHERE l.id = ?", (line,)).fetchone()[0]
        while parent[c][0] is not None:
            c = parent[c][0]
        return parent[c][1]

    for n in names:
        found = collections.Counter(root(first(n, q)) for q in ok if first(n, q) is not None and grades[q][first(n, q)] == 3)
        print(f"{n}, the sources it puts first: " + ", ".join(f"{c} {k}" for c, k in found.most_common()))

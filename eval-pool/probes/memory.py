"""A probe: a memory of the judged searches. A new search whose query vector (the app's int8 model) is within a
threshold of a judged search's gets that search's best good line first. Each judged search is tried against all the
others (leave one out); and the other evaluations' and the unjudged searches' queries against all of them."""
import csv, json, math, sys
from collections import Counter
import numpy as np
z = np.load("build/why/qvec-int8.npz"); QV = {str(k): v for k, v in zip(z["keys"], z["vectors"])}
status = {r["qid"]: r["status"] for r in csv.DictReader(open("judge/status.tsv", encoding="utf-8"), delimiter="\t")}
grades, src = {}, {}
for r in csv.DictReader(open("judge/grades.tsv", encoding="utf-8"), delimiter="\t"):
    grades.setdefault(r["qid"], {})[int(r["lineId"])] = int(r["grade"]); src[(r["qid"], int(r["lineId"]))] = r["from"]
key = {r["qid"]: r["query"] for r in csv.DictReader(open("build/judge-pick/key.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)}
runs = {}
for p in ("build/judge/run.jsonl", "build/demo/run.jsonl"):
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); runs.setdefault(r["query"], r)
ok = [q for q in grades if status.get(q, "ok") == "ok" and key[q] in runs and f"j/{q}" in QV]
def best_lines(q):
    """A judged search's lines to remember: its sources, else its good lines; a line of the page before one the judge brought."""
    good = [d for d, g in grades[q].items() if g >= 2]
    return sorted(good, key=lambda d: (-grades[q][d], src[(q, d)] != "page", d))
mem = [q for q in ok if best_lines(q)]
MV = np.stack([QV[f"j/{q}"] for q in mem])
def page_first(q):
    pg = [x["lineId"] for x in runs[key[q]]["hybrid"][:1]]
    return pg[0] if pg else None
out = {}
print(f"{len(ok)} judged searches, {len(mem)} of them with a good line to remember")
print("threshold | searches with another judged search that near | the remembered line: good / not good / not judged here | first good: the app's page -> with the memory (a line not judged here counted as not good, and as good)")
for tau in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
    c = Counter(); base = low = high = 0; pending = []
    for q in ok:
        sims = MV @ QV[f"j/{q}"]
        if q in mem: sims[mem.index(q)] = -1
        k = int(np.argmax(sims)); pf = page_first(q); g0 = grades[q].get(pf, -1) >= 2; base += g0
        if sims[k] < tau:
            low += g0; high += g0; continue
        r = best_lines(mem[k])[0]
        if r in grades[q]:
            g = grades[q][r] >= 2; c["good" if g else "not good"] += 1; low += g; high += g
            if not g: pending.append((q, mem[k], r, float(sims[k]), "judged-bad"))
        else:
            c["not judged"] += 1; high += 1; pending.append((q, mem[k], r, float(sims[k]), "unjudged"))
    n = c["good"] + c["not good"] + c["not judged"]
    print(f"  {tau:.2f} | {n:3d} | {c['good']} / {c['not good']} / {c['not judged']} | {base} -> {low} to {high}")
    out[str(tau)] = pending
json.dump(out, open("build/why/memory-pending.json", "w"))
# the other populations: how often the memory would speak
other = {}
for k in QV:
    name = k.split("/")[0]
    if name != "j": other.setdefault(name, []).append(k)
print("\nhow many searches of each other set have a judged search within the threshold (0.9 / 0.85 / 0.8 / 0.75):")
for name, ks in sorted(other.items()):
    best = np.array([float((MV @ QV[k]).max()) for k in ks])
    print(f"  {name:9s} {len(ks):5d}: " + " / ".join(str(int((best >= t).sum())) for t in (0.9, 0.85, 0.8, 0.75)))

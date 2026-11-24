"""Sefaria's division of the Bavli into passages, mapped to our lines, and what ranking by passage does on crfh1672:
the questions of five tractates, each with the one line that answers it.

    python eval-pool/probes/sefaria/sugyot.py build/rerank/vec-tune3-base.npz

The vectors are probes/vectors.py's over the pool (crfh's candidates are whole tractates).
"""
import collections, csv, json, math, pickle, re, sqlite3, sys
import bson, numpy as np
D = "build/sefaria/dump/dump/sefaria/"

def gematria(n):
    if n <= 0: return str(n)
    out = ""
    th, r = divmod(n, 1000)
    if th: out += gematria(th) + " "
    for v, l in ((400, "ת"), (300, "ש"), (200, "ר"), (100, "ק")):
        while r >= v: out += l; r -= v
    if r == 15: out += "טו"; r = 0
    elif r == 16: out += "טז"; r = 0
    for v, l in ((90, "צ"), (80, "פ"), (70, "ע"), (60, "ס"), (50, "נ"), (40, "מ"), (30, "ל"), (20, "כ"), (10, "י")):
        if r >= v: out += l; r -= v
    for v, l in ((9, "ט"), (8, "ח"), (7, "ז"), (6, "ו"), (5, "ה"), (4, "ד"), (3, "ג"), (2, "ב"), (1, "א")):
        if r >= v: out += l; r -= v
    return out

he_title = {}
with open(D + "index.bson", "rb") as f:
    for d in bson.decode_file_iter(f):
        s = d.get("schema") or {}
        if "nodes" not in s and (s.get("addressTypes") or [""])[0] == "Talmud":
            he_title[d["title"]] = next((t["text"] for t in s.get("titles", []) if t.get("lang") == "he" and t.get("primary")), None)
SEG = re.compile(r"^(.*?) (\d+)([ab]):(\d+)$")
con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
ours = {}
for line_id, he in con.execute("SELECT l.id, l.heRef FROM line l JOIN book b ON b.id = l.bookId WHERE b.sourceId = 1 AND l.heRef IS NOT NULL AND b.categoryId IN (SELECT descendantId FROM category_closure WHERE ancestorId IN (SELECT id FROM category WHERE title = 'תלמוד'))"):
    ours.setdefault(he, line_id)
print(len(ours), "referenced lines of the Talmud's books")
def line_of(ref):
    m = SEG.match(ref)
    if not m or m.group(1) not in he_title: return None
    return ours.get(f"{he_title[m.group(1)]} {gematria(int(m.group(2)))}{'.' if m.group(3) == 'a' else ':'}, {gematria(int(m.group(4)))}")

passage_of = {}                      # our line -> its passage (the order of the passage in the dump)
sizes = collections.Counter(); kinds = collections.Counter(); found = missed = 0
with open(D + "passage.bson", "rb") as f:
    for k, d in enumerate(bson.decode_file_iter(f)):
        if d.get("type") not in ("Sugya", "Mishnah"): continue
        lines = [line_of(r) for r in d.get("ref_list") or []]
        found += sum(l is not None for l in lines); missed += sum(l is None for l in lines)
        kinds[d["type"]] += 1
        for l in lines:
            if l is not None: passage_of[l] = k
print(dict(kinds), "| segments mapped", found, "| not mapped", missed, "| lines in a passage:", len(passage_of))

z = np.load(sys.argv[1], allow_pickle=True)
docs = dict(zip(z["doc_ids"].tolist(), z["docs"])); queries = dict(zip([str(k) for k in z["keys"]], z["queries"]))
spec = [s for s in json.load(open("build/rerank/pool-base.json"))["searches"] if s["key"].startswith("crfh/")]
right = {r["id"]: int(r["correct_line"]) for r in csv.DictReader(open("eval-crfh/lines.tsv", encoding="utf-8"), delimiter="\t")}
rows = []
cover = 0
for s in spec:
    qid = s["key"].split("/")[1]; correct = right[qid]
    cands = [d for d in s["candidates"] if d in docs]
    if correct not in docs: continue
    cover += correct in passage_of
    M = np.stack([docs[d] for d in cands]); sc = M @ queries[s["key"]]
    order = np.argsort(-sc, kind="stable")
    line_rank = int(np.where(order == cands.index(correct))[0][0]) + 1
    group = [passage_of.get(d, -d) for d in cands]             # a line outside every passage is a passage of its own
    best = collections.defaultdict(list)
    for g, v in zip(group, sc): best[g].append(float(v))
    def ranks(score_of):
        g_sc = {g: score_of(v) for g, v in best.items()}
        g_order = sorted(g_sc, key=lambda g: -g_sc[g])
        return g_order.index(passage_of.get(correct, -correct)) + 1, g_sc
    top1 = ranks(max)[0]
    top2 = ranks(lambda v: sum(sorted(v, reverse=True)[:2]) / min(2, len(v)))[0]
    # a line's own score with its passage's best line beside it
    out = {}
    for lam in (0.25, 0.5, 1.0):
        gmax = {g: max(v) for g, v in best.items()}
        mixed = np.array([v + lam * gmax[g] for g, v in zip(group, sc)])
        o2 = np.argsort(-mixed, kind="stable")
        out[lam] = int(np.where(o2 == cands.index(correct))[0][0]) + 1
    rows.append((line_rank, top1, top2, out, len(best), len(cands)))
n = len(rows)
print(f"{n} questions; the correct line is inside a passage in {cover}; passages a tractate (median): {sorted(r[4] for r in rows)[n//2]} of {sorted(r[5] for r in rows)[n//2]} lines")
def at(vals, k): return sum(v <= k for v in vals)
print("the correct LINE, by the line's own score:        first", at([r[0] for r in rows], 1), "| in 3", at([r[0] for r in rows], 3), "| in 10", at([r[0] for r in rows], 10))
print("the correct PASSAGE, by its best line:            first", at([r[1] for r in rows], 1), "| in 3", at([r[1] for r in rows], 3), "| in 10", at([r[1] for r in rows], 10))
print("the correct PASSAGE, by its two best lines:       first", at([r[2] for r in rows], 1), "| in 3", at([r[2] for r in rows], 3), "| in 10", at([r[2] for r in rows], 10))
for lam in (0.25, 0.5, 1.0):
    v = [r[3][lam] for r in rows]
    a = sum(x == 1 and r[0] != 1 for x, r in zip(v, rows)); b = sum(r[0] == 1 and x != 1 for x, r in zip(v, rows))
    print(f"the correct LINE, its score + {lam} x its passage's best: first {at(v, 1)} | in 3 {at(v, 3)} | in 10 {at(v, 10)} | won {a}, lost {b}, bar {2 * math.sqrt(a + b):.1f}")

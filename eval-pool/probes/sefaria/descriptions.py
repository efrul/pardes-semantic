"""The sources Sefaria's team wrote a Hebrew title and description for, as an outside test: does the pack's model find a
source by what was written about it?

    python eval-pool/probes/sefaria/descriptions.py [distractors]

Each described source is looked for among the described sources' own lines and random lines of the library's Sefaria
books, by its title, by its description, and by both.
"""
import collections, pickle, random, re, sqlite3, sys, time
import bson, numpy as np, torch
sys.path[:0] = [".", "eval-pool/probes/sefaria"]
import pool_rank
from embed_corpus import MIN_WORDS, strip_number
from pardes_semantic import model as m
from pardes_semantic.clean import clean
torch.set_num_threads(4)
D = "build/sefaria/dump/dump/sefaria/"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
MARKS = re.compile("[֑-ׇ]")
def norm(s):
    return " ".join(re.sub(r"[\"'׳״.,:;!?()\[\]\-–—<>/]", " ", MARKS.sub("", s).replace("־", " ")).split())

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
def primary(node, lang):
    return next((t["text"] for t in node.get("titles", []) if t.get("lang") == lang and t.get("primary")), "")
leaves = {}
with open(D + "index.bson", "rb") as f:
    for d in bson.decode_file_iter(f):
        s = d.get("schema") or {}
        en, he = d.get("title") or primary(s, "en"), primary(s, "he")
        if not en or not he: continue
        if "nodes" not in s:
            leaves[en] = (he + " ", s.get("addressTypes") or []); continue
        def walk(node, en_prefix, he_prefix):
            for c in node.get("nodes", []):
                default = c.get("default") or c.get("key") == "default"
                ce, ch = primary(c, "en"), primary(c, "he")
                e2 = en_prefix if default or not ce else en_prefix + ", " + ce
                h2 = he_prefix if default or not ch else he_prefix + ch + ", "
                if "nodes" in c: walk(c, e2, h2)
                else: leaves.setdefault(e2, (h2 + " ", c.get("addressTypes") or []))
        walk(s, en, he + ", ")
SECTIONS = re.compile(r"^(.*?) (\d+[ab]?(?::\d+[ab]?)*)$")
def our_ref(ref):
    mm = SECTIONS.match(ref)
    if not mm or mm.group(1) not in leaves: return None
    he_prefix, types = leaves[mm.group(1)]
    parts = []
    for i, sec in enumerate(mm.group(2).split(":")):
        kind = types[i] if i < len(types) else "Integer"
        if sec[-1] in "ab":
            if kind not in ("Talmud", "Folio"): return None
            parts.append(gematria(int(sec[:-1])) + ("." if sec[-1] == "a" else ":"))
        elif kind == "Talmud": return None
        else: parts.append(gematria(int(sec)))
    return he_prefix + ", ".join(parts)
con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
ours = {}
for line_id, he in con.execute("SELECT id, heRef FROM line WHERE heRef IS NOT NULL"):
    ours.setdefault(he, line_id)
items = []          # (title, description, origin, [our lines])
total = 0
with open(D + "topic_links.bson", "rb") as f:
    for d in bson.decode_file_iter(f):
        if d.get("class") != "refTopic" or d.get("is_sheet"): continue
        he = ((d.get("descriptions") or {}).get("he") or {})
        if not (he.get("title") or he.get("prompt")): continue
        total += 1
        lines = [ours[h] for h in (our_ref(r) for r in d.get("expandedRefs") or []) if h in ours]
        if lines:
            items.append((norm(he.get("title") or ""), norm(he.get("prompt") or ""), d.get("dataSource"), lines))
print(total, "described sources;", len(items), "of them on our lines;", collections.Counter(i[2] for i in items).most_common(), flush=True)
random.seed(77)
positives = {l for it in items for l in it[3]}
top = con.execute("SELECT MAX(id) FROM line").fetchone()[0]
pool = set()
while len(pool) < N * 3:
    pool.update(random.sample(range(1, top + 1), 5000))
ids = sorted(positives | pool)
texts = {}
kept_others = 0
for i in range(0, len(ids), 900):
    chunk = ids[i:i + 900]
    for line_id, content, he in con.execute(f"SELECT l.id, l.content, l.heRef FROM line l JOIN book b ON b.id = l.bookId WHERE b.sourceId = 1 AND l.id IN ({','.join('?' * len(chunk))})", chunk):
        if line_id not in positives and (he is None or kept_others >= N): continue
        t = strip_number(clean(content or ""))
        if len(t.split()) >= MIN_WORDS:
            texts[line_id] = m.DOCUMENT_PREFIX + t
            kept_others += line_id not in positives
print(len(texts), "lines to embed:", len(positives & set(texts)), "of the described sources and", kept_others, "others", flush=True)
loaded = m.load("cpu", torch.float32, sys.argv[2] if len(sys.argv) > 2 else "build/kaggle-output/tune3/tuned")
t0 = time.time()
vec = pool_rank.embed_docs(loaded, texts, 8192)
print("embedded in", round((time.time() - t0) / 60, 1), "min", flush=True)
docs = sorted(vec); M = np.stack([vec[d] for d in docs]); pos = {d: k for k, d in enumerate(docs)}
items = [it for it in items if any(l in pos for l in it[3])]
def run(name, texts_q, keep):
    its = [it for it, k in zip(items, keep) if k]
    Q = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + t for t, k in zip(texts_q, keep) if k])
    ranks = []
    for it, q in zip(its, Q):
        s = M @ q
        mine = [pos[l] for l in it[3] if l in pos]
        ranks.append(int((s > s[mine].max()).sum()) + 1)
    n = len(ranks); ranks.sort()
    print(f"{name:26s} {n:5d} | first {sum(r == 1 for r in ranks)} ({100*sum(r == 1 for r in ranks)/n:.0f}%) | in 10 {sum(r <= 10 for r in ranks)} ({100*sum(r <= 10 for r in ranks)/n:.0f}%) | in 100 {sum(r <= 100 for r in ranks)} ({100*sum(r <= 100 for r in ranks)/n:.0f}%) | median place {ranks[n//2]}", flush=True)
print(f"each source among {len(docs)} lines:")
run("by its title", [it[0] for it in items], [bool(it[0]) for it in items])
run("by its description", [it[1] for it in items], [bool(it[1]) for it in items])
run("by both", [(it[0] + ". " + it[1]).strip(". ") for it in items], [True] * len(items))
w = sorted(len(it[1].split()) for it in items if it[1]); tw = sorted(len(it[0].split()) for it in items if it[0])
print("words: title median", tw[len(tw)//2], "| description median", w[len(w)//2])
pickle.dump(items, open("build/sefaria/described.pkl", "wb"))

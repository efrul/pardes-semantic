"""How well the pack's model already finds a topic's source by the topic's name, and how near the users' searches are to
the topics."""
import collections, csv, pickle, random, re, sqlite3, sys, time
import bson, numpy as np, torch
sys.path[:0] = ["."]
import pool_rank
from embed_corpus import MIN_WORDS, strip_number
from pardes_semantic import model as m
from pardes_semantic.clean import clean
torch.set_num_threads(4)
D = "build/sefaria/dump/dump/sefaria/"
MARKS = re.compile("[֑-ׇ]")
def norm(s):
    return " ".join(re.sub(r"[\"'׳״.,:;!?()\[\]\-–—<>/]", " ", MARKS.sub("", s).replace("־", " ")).split())
pairs = pickle.load(open("build/sefaria/about_pairs.pkl", "rb"))
he, sub = {}, {}
with open(D + "topics.bson", "rb") as f:
    for t in bson.decode_file_iter(f):
        pr = [x["text"] for x in t.get("titles", []) if x.get("lang") == "he" and x.get("primary")] or [x["text"] for x in t.get("titles", []) if x.get("lang") == "he"]
        if pr: he[t["slug"]] = norm(pr[0])
        sub[t["slug"]] = t.get("subclass")
good = [p for p in pairs if p[0] in he and p[1] not in ("ashlag-glossary",) and sub.get(p[0]) not in ("person", "author")]
random.seed(9)
random.shuffle(good)
picked, seen = [], set()
for p in good:                      # one pair a topic first
    if p[0] not in seen:
        seen.add(p[0]); picked.append(p)
    if len(picked) == 1500: break
of_topic = collections.defaultdict(set)
for p in pairs:
    of_topic[p[0]].update(p[3])
positives = {p[3][0] for p in picked}
others = list({l for p in random.sample(pairs, 12000) for l in p[3][:1]} - positives)[:6500]
ids = sorted(positives | set(others))
con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
pool_rank.PREP = lambda t: strip_number(clean(t))
texts = {}
for i in range(0, len(ids), 900):
    chunk = ids[i:i + 900]
    for line_id, content in con.execute(f"SELECT id, content FROM line WHERE id IN ({','.join('?' * len(chunk))})", chunk):
        t = pool_rank.PREP(content or "")
        if len(t.split()) >= MIN_WORDS:
            texts[line_id] = m.DOCUMENT_PREFIX + t
loaded = m.load("cpu", torch.float32, "build/kaggle-output/tune3/tuned")
t0 = time.time()
vec = pool_rank.embed_docs(loaded, texts, 8192)
print(len(vec), "lines embedded in", round((time.time() - t0) / 60, 1), "min", flush=True)
docs = sorted(vec); M = np.stack([vec[d] for d in docs]); pos = {d: k for k, d in enumerate(docs)}
picked = [p for p in picked if p[3][0] in vec]
Q = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + he[p[0]] for p in picked])
def toks(line):
    t = norm(line).split()
    return set(t) | {x[1:] for x in t if len(x) > 3 and x[0] in "הבלמוכשד"} | {x[2:] for x in t if len(x) > 4}
ranks = collections.defaultdict(list)
for p, q in zip(picked, Q):
    s = M @ q
    mine = pos[p[3][0]]
    same = [pos[l] for l in of_topic[p[0]] if l in pos and pos[l] != mine]
    s2 = s.copy(); s2[same] = -9
    rank = int((s2 > s[mine]).sum()) + 1
    words = [w for w in he[p[0]].split() if len(w) > 1 and w not in ("של", "על", "את", "דיני", "הלכות")]
    share = sum(w in toks(texts[p[3][0]]) for w in words) / max(1, len(words))
    kind = "the line holds the topic's words" if share >= 1 else ("some of them" if share > 0 else "none of them")
    ranks[kind].append(rank); ranks["all"].append(rank); ranks["origin " + p[1]].append(rank)
print(f"the topic's own source among {len(docs)} lines, by the topic's name as the query:")
for kind, v in sorted(ranks.items(), key=lambda x: -len(x[1])):
    v.sort(); n = len(v)
    print(f"  {kind:36s} {n:5d} | first {100*sum(x == 1 for x in v)/n:3.0f}% | in ten {100*sum(x <= 10 for x in v)/n:3.0f}% | in 100 {100*sum(x <= 100 for x in v)/n:3.0f}% | median place {v[n//2]}")
# the users' searches against the topics' names
titles = sorted({he[s] for s in of_topic if s in he and sub.get(s) not in ("person", "author")})
T = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + t for t in titles])
rows = list(csv.DictReader(open("build/feedback/history.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
qs = sorted({norm(r["query"]) for r in rows if r.get("query") and norm(r["query"])})
S = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + q for q in qs])
best = (S @ T.T).max(1)
arg = (S @ T.T).argmax(1)
print(f"{len(qs)} searches of the history against {len(titles)} topic names: nearest topic at cosine", {th: int((best >= th).sum()) for th in (0.95, 0.9, 0.85, 0.8, 0.7)})
by = collections.defaultdict(list)
for q, b in zip(qs, best): by[min(len(q.split()), 6)].append(float(b))
print("  by words, share at 0.85 or more:", {k: f"{100*sum(x >= .85 for x in v)/len(v):.0f}% of {len(v)}" for k, v in sorted(by.items())})
np.savez("build/sefaria/near.npz", qs=np.array(qs), best=best, title=np.array([titles[a] for a in arg]))

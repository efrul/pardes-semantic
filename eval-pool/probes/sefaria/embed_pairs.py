"""The pack's model's vectors of a sample of Sefaria's topic pairs: the topics' names as queries, their sources' lines."""
import collections, json, pickle, random, re, sqlite3, sys, time
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
    return " ".join(MARKS.sub("", s).replace("־", " ").split())
PER_TOPIC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
pairs = pickle.load(open("build/sefaria/about_pairs.pkl", "rb"))
he = {}
with open(D + "topics.bson", "rb") as f:
    for t in bson.decode_file_iter(f):
        pr = [x["text"] for x in t.get("titles", []) if x.get("lang") == "he" and x.get("primary")] or [x["text"] for x in t.get("titles", []) if x.get("lang") == "he"]
        if pr: he[t["slug"]] = norm(pr[0])
# no line an evaluation ranks: the pool's candidates (the five tractates of crfh whole) and the judged searches' 150
held = set()
for path in ("build/rerank/pool-base.json", "build/why/spec-know.json"):
    for s in json.load(open(path))["searches"]:
        held.update(s["candidates"])
by_topic = collections.defaultdict(list)
for topic, src, ref, lines, cp, tfidf, pr in pairs:
    if src == "ashlag-glossary" or topic not in he or lines[0] in held:
        continue
    by_topic[topic].append((lines[0], src))
random.seed(613)
chosen = []
for topic, rows in sorted(by_topic.items()):
    random.shuffle(rows)
    seen = set()
    for line, src in rows:
        if line not in seen:
            seen.add(line); chosen.append((topic, line, src))
        if len(seen) == PER_TOPIC: break
print(len(chosen), "pairs of", len(by_topic), "topics;", len({c[1] for c in chosen}), "lines;", len(held), "lines held out", flush=True)
ids = sorted({c[1] for c in chosen})
con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
texts = {}
for i in range(0, len(ids), 900):
    chunk = ids[i:i + 900]
    for line_id, content in con.execute(f"SELECT id, content FROM line WHERE id IN ({','.join('?' * len(chunk))})", chunk):
        t = strip_number(clean(content or ""))
        if len(t.split()) >= MIN_WORDS:
            texts[line_id] = m.DOCUMENT_PREFIX + t
loaded = m.load("cpu", torch.float32, "build/kaggle-output/tune3/tuned")
t0 = time.time()
vec = pool_rank.embed_docs(loaded, texts, 8192)
print(len(vec), "lines embedded in", round((time.time() - t0) / 60, 1), "min", flush=True)
topics = sorted({c[0] for c in chosen})
T = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + he[t] for t in topics])
docs = sorted(vec)
np.savez("build/sefaria/pairs-tune3.npz", topics=np.array(topics), topic_vecs=T.astype(np.float32), doc_ids=np.array(docs, dtype=np.int64),
         docs=np.stack([vec[d] for d in docs]).astype(np.float32),
         pair_topic=np.array([c[0] for c in chosen if c[1] in vec]), pair_line=np.array([c[1] for c in chosen if c[1] in vec], dtype=np.int64),
         pair_origin=np.array([c[2] for c in chosen if c[1] in vec]))
print("-> build/sefaria/pairs-tune3.npz")

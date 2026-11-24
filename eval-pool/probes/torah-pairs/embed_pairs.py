"""The pack's model's vectors of a sample of Torah-pairs: each query and the line of ours its passage points at."""
import collections, json, pickle, random, re, sqlite3, sys, time
import numpy as np, torch
sys.path[:0] = ["."]
import pool_rank
from embed_corpus import MIN_WORDS, strip_number
from pardes_semantic import model as m
from pardes_semantic.clean import clean
torch.set_num_threads(4)
MARKS = re.compile("[֑-ׇ]")
def norm(s):
    return " ".join(MARKS.sub("", s or "").replace("־", " ").split())
TAKE = {"torahqa.clean": 12000, "micropedia": 5000, "et": 5000, "shut_passages": 4000}
mapped = pickle.load(open("build/torah-pairs/mapped.pkl", "rb"))
held = set()
for path in ("build/rerank/pool-base.json", "build/why/spec-know.json"):
    for s in json.load(open(path))["searches"]:
        held.update(s["candidates"])
random.seed(1729)
chosen = []
for name, n in TAKE.items():
    rows = [r for r in mapped[name] if r[2] not in held and len(norm(r[1]).split()) >= 3 and r[3] in ("high", "exact", "strong")]
    random.shuffle(rows)
    seen_q, seen_l, got = set(), set(), 0
    for r in rows:
        q = norm(r[1])
        if q in seen_q or r[2] in seen_l: continue      # a line once, a question once
        seen_q.add(q); seen_l.add(r[2]); chosen.append((name, q, r[2])); got += 1
        if got == n: break
    print(name, "usable", len(rows), "taken", got, flush=True)
ids = sorted({c[2] for c in chosen})
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
chosen = [c for c in chosen if c[2] in vec]
Q = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + c[1] for c in chosen])
docs = sorted(vec)
np.savez("build/torah-pairs/pairs-tune3.npz", queries=Q.astype(np.float32), source=np.array([c[0] for c in chosen]),
         line=np.array([c[2] for c in chosen], dtype=np.int64), doc_ids=np.array(docs, dtype=np.int64),
         docs=np.stack([vec[d] for d in docs]).astype(np.float32))
print(len(chosen), "pairs -> build/torah-pairs/pairs-tune3.npz", flush=True)

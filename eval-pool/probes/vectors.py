"""A probe: a model's vectors of a pool spec's queries and candidates, as pool_rank.py embeds them (merge + strip),
kept as .npz for experiments on top of the frozen vectors.

    python eval-pool/probes/vectors.py <model dir> <spec.json> <out.npz>
"""
import json, sqlite3, sys
import numpy as np, torch
sys.path[:0] = ["."]
import pool_rank
from embed_corpus import MIN_WORDS, strip_number
from pardes_semantic import model as m
from pardes_semantic.clean import clean

torch.set_num_threads(4)
source, spec, out = sys.argv[1:4]
pool_rank.PREP = lambda t: strip_number(clean(t))
searches = json.load(open(spec, encoding="utf-8"))["searches"]
ids = sorted({d for s in searches for d in s["candidates"]})
con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
texts = {}
for i in range(0, len(ids), 900):
    chunk = ids[i:i + 900]
    for line_id, content in con.execute(f"SELECT id, content FROM line WHERE id IN ({','.join('?' * len(chunk))})", chunk):
        t = pool_rank.PREP(content or "")
        if len(t.split()) >= MIN_WORDS:
            texts[line_id] = m.DOCUMENT_PREFIX + t
texts = pool_rank.short_context(con, texts, "merge", 8)
loaded = m.load("cpu", torch.float32, source)
vec = pool_rank.embed_docs(loaded, texts, 8192)
qv = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + s["query"] for s in searches])
docs = sorted(vec)
np.savez(out, doc_ids=np.array(docs, dtype=np.int64), docs=np.stack([vec[d] for d in docs]).astype(np.float32),
         keys=np.array([s["key"] for s in searches]), queries=np.asarray(qv, dtype=np.float32))
print(len(docs), "lines,", len(searches), "queries ->", out)

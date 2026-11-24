"""A probe: another family's embedder (pool_rank.py --st) under other prompts, to tell a model's place from the place
of the prompt it was asked with. The lines are embedded once for each prompt of theirs, the queries for each of theirs.

    python eval-pool/probes/st_prompts.py <model> <spec.json> <out dir> <key prefix> name=query prompt[||line prompt] ...

Run from the repository's root. A prompt is its text ("task: question answering | query: "), empty for none; without
"||" the lines get the model's own "document" prompt. Of the spec's searches whose key starts with the prefix
("crfh/"), with the lines as pool_rank.py embeds them (merge + strip). Writes <out dir>/ranks-<name>.json for
eval-crfh/measure.py and ../judged_measure.py.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["."]
import hub_score
import pool_rank
from embed_corpus import MIN_WORDS, strip_number
from pardes_semantic import model as m
from pardes_semantic.clean import clean
from sentence_transformers import SentenceTransformer

model, spec, out, prefix = sys.argv[1:5]
variants = [a.partition("=")[::2] for a in sys.argv[5:]]
searches = [s for s in json.load(open(spec, encoding="utf-8"))["searches"] if s["key"].startswith(prefix)]
ids = sorted({d for s in searches for d in s["candidates"]})
con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
pool_rank.PREP = lambda t: strip_number(clean(t))
texts = {}
for i in range(0, len(ids), 900):
    chunk = ids[i:i + 900]
    for line_id, content in con.execute(f"SELECT id, content FROM line WHERE id IN ({','.join('?' * len(chunk))})", chunk):
        t = pool_rank.PREP(content or "")
        if len(t.split()) >= MIN_WORDS:
            texts[line_id] = m.DOCUMENT_PREFIX + t
texts = pool_rank.short_context(con, texts, "merge", 8)
docs = sorted(texts)
lines = [hub_score.plain(texts[d][len(m.DOCUMENT_PREFIX):]) for d in docs]
queries = [hub_score.plain(s["query"]) for s in searches]
text_only = {"config_kwargs": {"vision_config": None, "audio_config": None}} if "embeddinggemma-2" in model.lower() else {}
st = SentenceTransformer(model, device="cpu", **text_only)


def embed(items: list[str], prompt: str, batch: int, max_tokens: int) -> np.ndarray:
    st.max_seq_length = max_tokens
    order = sorted(range(len(items)), key=lambda k: len(items[k]))
    vectors = [None] * len(items)
    for i in range(0, len(order), batch):
        chunk = order[i:i + batch]
        found = st.encode([items[k] for k in chunk], batch_size=len(chunk), normalize_embeddings=True,
                          prompt=prompt or None, show_progress_bar=False)
        for k, v in zip(chunk, found):
            vectors[k] = v
    return np.asarray(vectors, dtype=np.float32)


embedded = {}
for name, prompts in variants:
    query_prompt, sep, line_prompt = prompts.partition("||")
    if not sep:
        line_prompt = st.prompts["document"]
    if line_prompt not in embedded:
        embedded[line_prompt] = dict(zip(docs, embed(lines, line_prompt, 8, m.DOCUMENT_MAX_TOKENS)))
        print(f"{len(docs)} lines after {line_prompt!r}", flush=True)
    qv = embed(queries, query_prompt, 32, m.QUERY_MAX_TOKENS)
    print(f"{name}: the queries after {query_prompt!r}", flush=True)
    pool_rank.write_ranks(searches, embedded[line_prompt], qv, Path(out) / f"ranks-{name}.json")

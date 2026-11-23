"""A probe's input: the query vectors as the app computes them (the pack's int8 model), of the judged searches, of the
pool's other searches and of the unjudged searches, for memory.py and the split by how near a search is to another.

    python eval-pool/probes/query_vectors.py

Run from the repository's root; writes build/why/qvec-int8.npz (keys, vectors).
"""
import csv
import json
import sys
import time
from collections import Counter

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path[:0] = ["."]
from pardes_semantic.clean import clean  # noqa: E402

model = "build/library/pardes-model"
spec = json.load(open(f"{model}/pardes-model.json", encoding="utf-8"))
tok = Tokenizer.from_file(f"{model}/{spec['tokenizer']}")
tok.enable_truncation(spec["queryMaxTokens"])
sess = ort.InferenceSession(f"{model}/{spec['onnx']}", providers=["CPUExecutionProvider"])
names = {i.name for i in sess.get_inputs()}


def embed(text):
    e = tok.encode(spec["queryPrefix"] + clean(text))
    feed = {"input_ids": np.array([e.ids], dtype=np.int64), "attention_mask": np.array([e.attention_mask], dtype=np.int64)}
    v = sess.run([spec["output"]], {k: x for k, x in feed.items() if k in names})[0][0]
    return (v / np.linalg.norm(v)).astype(np.float32)


queries = {}
for r in csv.DictReader(open("build/judge-pick/key.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
    queries[f"j/{r['qid']}"] = r["query"]
for s in json.load(open("build/judged/pool.json", encoding="utf-8"))["searches"]:
    if s["key"].split("/")[0] in ("real", "v1", "feedback", "crfh", "dev", "e100"):
        queries[s["key"]] = s["query"]
for s in json.load(open("build/distill/pool.json", encoding="utf-8"))["searches"]:
    queries[s["key"]] = s["query"]
t0 = time.time()
keys = sorted(queries)
V = np.stack([embed(queries[k]) for k in keys])
np.savez("build/why/qvec-int8.npz", keys=np.array(keys), vectors=V)
print(len(keys), "queries in", round(time.time() - t0), "s", dict(Counter(k.split("/")[0] for k in keys)))

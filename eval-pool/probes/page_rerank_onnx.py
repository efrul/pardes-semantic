"""A reranker's ONNX file on the CPU, over the first lines of the judged searches' pages: how long a page takes and
whether the order holds (page_rerank.py measures the ranks file it writes).

    python eval-pool/probes/page_page_rerank_onnx.py --model model.onnx --tokenizer tokenizer.json --out ranks.json [--top 6] [--max-tokens 384]
                          [--threads 4] [--one-by-one]

Run from the repository's root. A search's candidates are the page's first --top lines (pack 4's page, as the judges
saw it) and, after them, the other judged lines (the sources the judges knew): the file's order is the reranker's over
all of them, and the time printed is for the page's lines alone, a search at a time.
"""
import argparse
import csv
import json
import sqlite3
import sys
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path.insert(0, ".")
import hub_score  # noqa: E402
from embed_corpus import MIN_WORDS, strip_number  # noqa: E402
from pardes_semantic.clean import clean  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=6)
ap.add_argument("--max-tokens", type=int, default=384)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--one-by-one", action="store_true", help="a pair at a time, no padding")
args = ap.parse_args()

status = {r["qid"]: r["status"] for r in csv.DictReader(open("judge/status.tsv", encoding="utf-8"), delimiter="\t")}
key = {r["qid"]: r["query"] for r in csv.DictReader(open("build/judge-pick/key.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)}
grades = {}
for r in csv.DictReader(open("judge/grades.tsv", encoding="utf-8"), delimiter="\t"):
    grades.setdefault(r["qid"], {})[int(r["lineId"])] = int(r["grade"])
runs = {}
for p in ("build/judge/run.jsonl", "build/demo/run.jsonl"):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        runs.setdefault(r["query"], r)
ok = [q for q in grades if status.get(q, "ok") == "ok" and key[q] in runs]
if args.limit:
    ok = ok[:args.limit]
ids = sorted({d for q in ok for d in grades[q]})
con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
text = {}
for i in range(0, len(ids), 900):
    chunk = ids[i:i + 900]
    for line_id, content in con.execute(f"SELECT id, content FROM line WHERE id IN ({','.join('?' * len(chunk))})", chunk):
        t = hub_score.plain(strip_number(clean(content or "")))
        if len(t.split()) >= MIN_WORDS:
            text[line_id] = t

tok = Tokenizer.from_file(args.tokenizer)
tok.enable_truncation(max_length=args.max_tokens, strategy="only_second")
pad = tok.token_to_id("<pad>") if tok.token_to_id("<pad>") is not None else 0
so = ort.SessionOptions()
so.intra_op_num_threads = args.threads
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
t0 = time.time()
sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
names = {i.name for i in sess.get_inputs()}
print(f"loaded in {time.time() - t0:.1f} s; inputs {sorted(names)}", flush=True)


def score(query, lines):
    if not lines:
        return []
    enc = [tok.encode(query, t) for t in lines]
    groups = [[e] for e in enc] if args.one_by_one else [enc]
    out = []
    for g in groups:
        n = max(len(e.ids) for e in g)
        ids_ = np.full((len(g), n), pad, dtype=np.int64)
        mask = np.zeros((len(g), n), dtype=np.int64)
        for k, e in enumerate(g):
            ids_[k, :len(e.ids)] = e.ids
            mask[k, :len(e.ids)] = 1
        feed = {"input_ids": ids_, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids_)
        out.extend(sess.run(None, feed)[0].reshape(-1).tolist())
    return out


def page(q):
    return [x["lineId"] for x in runs[key[q]]["hybrid"][:args.top] if x["lineId"] in grades[q] and x["lineId"] in text]


score(clean(key[ok[0]]), [text[d] for d in page(ok[0])])  # warm up
ranks, times, tokens = {}, [], []
for n, q in enumerate(ok):
    query = hub_score.plain(clean(key[q]))
    p = page(q)
    t = time.time()
    s = score(query, [text[d] for d in p])
    times.append(time.time() - t)
    tokens.append(sum(len(tok.encode(query, text[d]).ids) for d in p))
    rest = [d for d in grades[q] if d in text and d not in p]
    s += score(query, [text[d] for d in rest])
    order = sorted(zip(p + rest, s), key=lambda x: -x[1])
    ranks[f"j/{q}"] = {"ids": [d for d, _ in order], "scores": [round(x, 5) for _, x in order]}
    if n % 100 == 0:
        print(f"  {n}/{len(ok)}, {sum(times):.0f} s on the pages", flush=True)
json.dump(ranks, open(args.out, "w"))
times.sort()
print(f"{len(ok)} searches, the page's first {args.top}: mean {1000 * sum(times) / len(times):.0f} ms, median "
      f"{1000 * times[len(times) // 2]:.0f}, p90 {1000 * times[int(len(times) * .9)]:.0f}, max {1000 * times[-1]:.0f}; "
      f"tokens a page: mean {sum(tokens) / len(tokens):.0f}, max {max(tokens)} -> {args.out}")

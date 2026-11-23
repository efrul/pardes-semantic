"""A probe: a small reranker (a cross-encoder that reads the query and the line together) fine-tuned on the judges'
grades of two thirds of the judged searches, and tried on the third it did not see. The embedding model memorized such
a training and carried nothing to other searches; does a reranker, which learns what an answer looks like and not
where a search's vector should point, carry more?

    python eval-pool/probes/rerank_tune.py [--model Horizon-Labs/multilingual-reranker-small] [--epochs 3] [--lr 2e-5]
                              [--max-tokens 256] [--out build/rerank/tuned-grades]

Run from the repository's root, on the CPU. Trains on the searches of group.json's half and other-half, a search's
judged lines a step (the page's six of pack 4 and the sources the judges knew); after each epoch reports the first
judged line good over the page's six, for the evaluation third and by how near each of its searches is to a training
search (the cosine of the released model's query vectors), and for the training searches themselves.
"""
import argparse
import csv
import json
import math
import random
import sqlite3
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, ".")
import hub_score  # noqa: E402
from embed_corpus import MIN_WORDS, strip_number  # noqa: E402
from pardes_semantic.clean import clean  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Horizon-Labs/multilingual-reranker-small")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--most", type=int, default=10, help="a training step's lines at most")
ap.add_argument("--accumulate", type=int, default=4)
ap.add_argument("--train-groups", default="half,other-half")
ap.add_argument("--out", default="build/rerank/tuned-grades")
ap.add_argument("--eval-only", action="store_true")
args = ap.parse_args()
random.seed(5786)
torch.manual_seed(5786)
torch.set_num_threads(4)

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
group = json.load(open("build/why/group.json"))
ok = [q for q in grades if status.get(q, "ok") == "ok" and key[q] in runs]
ids = sorted({d for q in ok for d in grades[q]})
con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
text = {}
for i in range(0, len(ids), 900):
    chunk = ids[i:i + 900]
    for line_id, content in con.execute(f"SELECT id, content FROM line WHERE id IN ({','.join('?' * len(chunk))})", chunk):
        t = hub_score.plain(strip_number(clean(content or "")))
        if len(t.split()) >= MIN_WORDS:
            text[line_id] = t
train_groups = set(args.train_groups.split(","))
train_q = [q for q in ok if group.get(q) in train_groups]
eval_q = [q for q in ok if group.get(q) == "eval"]
rest_q = [q for q in ok if group.get(q) not in train_groups and group.get(q) != "eval"]

# how near each search is to a training search, by the released model's query vectors
z = np.load("build/why/qvec-int8.npz")
qv = {str(k).split("/")[1]: v for k, v in zip(z["keys"], z["vectors"] if "vectors" in z else z["queries"]) if str(k).startswith("j/")}
T = np.stack([qv[q] for q in train_q if q in qv])
near = {q: float((T @ qv[q]).max()) for q in eval_q + rest_q if q in qv}

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model, torch_dtype=torch.float32)


def page(q):
    return [x["lineId"] for x in runs[key[q]]["hybrid"][:6] if x["lineId"] in grades[q] and x["lineId"] in text]


def encode(q, lines):
    query = hub_score.plain(clean(key[q]))
    return tok([query] * len(lines), [text[d] for d in lines], padding=True, truncation="only_second",
               max_length=args.max_tokens, return_tensors="pt")


@torch.no_grad()
def scores(q, lines):
    model.eval()
    out = []
    for i in range(0, len(lines), 8):
        out.extend(model(**encode(q, lines[i:i + 8])).logits[:, 0].tolist())
    return out


def measure(qs, label):
    first, three, full = {}, 0, 0
    for q in qs:
        p = page(q)
        rest = [d for d in grades[q] if d in text and d not in p]
        s = dict(zip(p + rest, scores(q, p + rest)))
        o = sorted(p, key=lambda d: -s[d])
        first[q] = bool(o) and grades[q][o[0]] >= 2
        three += any(grades[q][d] >= 2 for d in o[:3])
        a = sorted(p + rest, key=lambda d: -s[d])
        full += bool(a) and grades[q][a[0]] >= 2
    base = {q: bool(page(q)) and grades[q][page(q)[0]] >= 2 for q in qs}
    x = sum(base[q] and not first[q] for q in qs)
    y = sum(first[q] and not base[q] for q in qs)
    line = (f"{label}: {len(qs)} searches; the page {sum(base.values())}; reranked: first good {sum(first.values())} "
            f"(only the page {x}, only reranked {y}, bar {2 * math.sqrt(x + y):.1f}), in three {three}, with the sources {full}")
    if all(q in near for q in qs):
        for lo, hi, name in ((0.85, 9, "a training search within 0.85"), (0.75, 0.85, "0.75 to 0.85"), (-9, 0.75, "none within 0.75")):
            sel = [q for q in qs if lo <= near[q] < hi]
            line += f"\n      {name}: {len(sel)}; the page {sum(base[q] for q in sel)}, reranked {sum(first[q] for q in sel)}"
    print(line, flush=True)
    return first


t0 = time.time()
print(f"{len(train_q)} training searches, {len(eval_q)} of the evaluation, {len(rest_q)} others; {args.model}", flush=True)
before = {"eval": measure(eval_q, "as the reranker is, evaluation")}
if rest_q:
    before["rest"] = measure(rest_q, "as the reranker is, the other searches")
before["train"] = measure(train_q, "as the reranker is, training")
print(f"   measured in {(time.time() - t0) / 60:.1f} min", flush=True)
if args.eval_only:
    raise SystemExit

P = {0: 0.05, 1: 0.25, 2: 0.8, 3: 0.97}
steps = args.epochs * len(train_q) // args.accumulate
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / steps))
done = 0
for epoch in range(args.epochs):
    random.shuffle(train_q)
    model.train()
    total, t1 = 0.0, time.time()
    for n, q in enumerate(train_q):
        lines = [d for d in grades[q] if d in text]
        goods = [d for d in lines if grades[q][d] >= 2]
        others = [d for d in lines if grades[q][d] < 2]
        random.shuffle(goods)
        random.shuffle(others)
        lines = (goods[:max(1, args.most // 2)] + others)[:args.most]
        if len(lines) < 2:
            continue
        g = torch.tensor([grades[q][d] for d in lines], dtype=torch.float32)
        logits = model(**encode(q, lines)).logits[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([P[int(x)] for x in g]))
        if (g >= 2).any() and (g < 2).any():
            target = torch.softmax(g * 2.0, dim=0)  # the good lines share the mass, the source more
            loss = loss + -(target * torch.log_softmax(logits, dim=0)).sum()
        (loss / args.accumulate).backward()
        total += float(loss.detach())
        if (n + 1) % args.accumulate == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            done += 1
        if n % 50 == 0:
            print(f"   epoch {epoch + 1}, {n}/{len(train_q)}, loss {total / (n + 1):.3f}, {(time.time() - t1) / 60:.1f} min", flush=True)
    after = measure(eval_q, f"after epoch {epoch + 1}, evaluation")
    x = sum(before["eval"][q] and not after[q] for q in eval_q)
    y = sum(after[q] and not before["eval"][q] for q in eval_q)
    print(f"      against the reranker as it was: only as it was {x}, only tuned {y}, bar {2 * math.sqrt(x + y):.1f}", flush=True)
    if rest_q:
        measure(rest_q, f"after epoch {epoch + 1}, the other searches")
measure(train_q, "after the training, training")
model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print(f"-> {args.out}; {(time.time() - t0) / 60:.0f} min")

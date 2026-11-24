"""A linear map on the pack's model's frozen vectors, learnt from Sefaria's topic pairs (a topic's name -> a line of one
of its sources), and what it does to our own evaluations: does this data teach something that carries to real searches?

A cheap stand-in for training the model on the pairs: one matrix over the vectors, queries' and lines' alike
(x + xW, as probes/projection.py), in-batch negatives. A tenth of the topics is kept out to see the map on topics it
did not learn.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np, torch
sys.path[:0] = ["."]
import pool_rank

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="build/sefaria/pairs-tune3.npz")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--decay", type=float, default=1e-2)
ap.add_argument("--temp", type=float, default=0.05)
ap.add_argument("--out", default="build/sefaria/ranks")
ap.add_argument("--origins", default="", help="comma separated: only pairs of these origins")
args = ap.parse_args()
torch.manual_seed(5786); torch.set_num_threads(4)
z = np.load(args.pairs, allow_pickle=True)
tvec = dict(zip([str(t) for t in z["topics"]], torch.tensor(z["topic_vecs"])))
dvec = dict(zip(z["doc_ids"].tolist(), torch.tensor(z["docs"])))
keep = set(args.origins.split(",")) if args.origins else None
pairs = [(str(t), int(l)) for t, l, o in zip(z["pair_topic"], z["pair_line"], z["pair_origin"]) if keep is None or str(o) in keep]
topics = sorted({t for t, _ in pairs})
held = {t for k, t in enumerate(topics) if k % 10 == 0}
train = [p for p in pairs if p[0] not in held]
test = [p for p in pairs if p[0] in held]
print(f"{len(pairs)} pairs of {len(topics)} topics: {len(train)} to learn from, {len(test)} of {len(held)} topics kept out")
dim = z["docs"].shape[1]

def in_batch(w, rows, batch=512):
    """Of each pair, is its own line the nearest of the batch's lines to its topic?"""
    hit = n = 0
    g = torch.Generator().manual_seed(1)
    order = torch.randperm(len(rows), generator=g).tolist()
    for i in range(0, len(order) - batch + 1, batch):
        chunk = [rows[k] for k in order[i:i + batch]]
        q = torch.stack([tvec[t] for t, _ in chunk]); d = torch.stack([dvec[l] for _, l in chunk])
        if w is not None:
            q, d = q + q @ w, d + d @ w
        q, d = torch.nn.functional.normalize(q, dim=1), torch.nn.functional.normalize(d, dim=1)
        s = q @ d.T
        same = torch.tensor([[a[0] == b[0] for b in chunk] for a in chunk])
        s = s.masked_fill(same & ~torch.eye(len(chunk), dtype=torch.bool), -9)   # another line of the same topic is no negative
        hit += int((s.argmax(1) == torch.arange(len(chunk))).sum()); n += len(chunk)
    return hit / max(1, n)

w = torch.nn.Parameter(torch.zeros(dim, dim))
opt = torch.optim.AdamW([w], lr=args.lr, weight_decay=args.decay)
print(f"before: its own line first among {args.batch}: learnt topics {in_batch(None, train):.3f}, topics kept out {in_batch(None, test):.3f}")
for step in range(1, args.steps + 1):
    idx = torch.randint(len(train), (args.batch,)).tolist()
    chunk = [train[k] for k in idx]
    q = torch.stack([tvec[t] for t, _ in chunk]); d = torch.stack([dvec[l] for _, l in chunk])
    q = torch.nn.functional.normalize(q + q @ w, dim=1); d = torch.nn.functional.normalize(d + d @ w, dim=1)
    s = (q @ d.T) / args.temp
    same = torch.tensor([[a[0] == b[0] for b in chunk] for a in chunk]) & ~torch.eye(len(chunk), dtype=torch.bool)
    loss = torch.nn.functional.cross_entropy(s.masked_fill(same, -1e4), torch.arange(len(chunk)))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == args.steps:
        with torch.no_grad():
            print(f"step {step}: loss {loss.item():.3f} | learnt {in_batch(w.detach(), train):.3f} | kept out {in_batch(w.detach(), test):.3f}", flush=True)
W = w.detach().numpy()

def apply(v):
    y = v + v @ W
    return y / np.linalg.norm(y, axis=-1, keepdims=True)

Path(args.out).mkdir(parents=True, exist_ok=True)
for name, vectors, spec in (("base", "build/rerank/vec-tune3-base.npz", "build/rerank/pool-base.json"),
                            ("know", "build/rerank/vec-tune3-know.npz", "build/why/spec-know.json")):
    v = np.load(vectors, allow_pickle=True)
    searches = json.loads(Path(spec).read_text("utf-8"))["searches"]
    qv = dict(zip([str(k) for k in v["keys"]], v["queries"]))
    searches = [s for s in searches if s["key"] in qv]
    Q = np.stack([qv[s["key"]] for s in searches])
    docs = dict(zip(v["doc_ids"].tolist(), v["docs"]))
    pool_rank.write_ranks(searches, docs, Q, Path(args.out) / f"ranks-tune3-{name}.json")
    mapped = dict(zip(v["doc_ids"].tolist(), apply(v["docs"])))
    pool_rank.write_ranks(searches, mapped, apply(Q), Path(args.out) / f"ranks-topics-{name}.json")

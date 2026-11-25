"""A linear map on the pack's model's frozen vectors, learnt from the private collection (a question or a claim -> the
line of ours that answers it), and what it does to questions and lines it did not see and to our own evaluations.

One matrix over the vectors, queries' and lines' alike (x + xW), in-batch negatives, as probes/sefaria/topic_map.py. A
tenth of the pairs is kept out: their questions and their lines are not learnt from.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np, torch
sys.path[:0] = ["."]
import pool_rank
ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="build/collection/pairs-tune3.npz")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--decay", type=float, default=1e-2)
ap.add_argument("--temp", type=float, default=0.05)
ap.add_argument("--sources", default="", help="comma separated: learn only from these files' pairs")
ap.add_argument("--out", default="build/collection/ranks")
args = ap.parse_args()
torch.manual_seed(5786); torch.set_num_threads(4)
z = np.load(args.pairs, allow_pickle=True)
Q = torch.tensor(z["queries"]); src = [str(s) for s in z["source"]]; line = z["line"].tolist()
pos = {d: k for k, d in enumerate(z["doc_ids"].tolist())}
Dm = torch.tensor(z["docs"])
D = Dm[[pos[l] for l in line]]
n = len(line)
test = np.array([k % 10 == 0 for k in range(n)])
keep = set(args.sources.split(",")) if args.sources else None
train_idx = [k for k in range(n) if not test[k] and (keep is None or src[k] in keep)]
test_idx = [k for k in range(n) if test[k]]
print(f"{n} pairs: {len(train_idx)} to learn from, {len(test_idx)} kept out")

def place(w, idx):
    """Each kept pair's own line among all the lines of the sample: first, in ten, the median place."""
    q, d = Q[idx], Dm
    if w is not None:
        q, d = q + q @ w, d + d @ w
    q, d = torch.nn.functional.normalize(q, dim=1), torch.nn.functional.normalize(d, dim=1)
    s = q @ d.T
    own = torch.tensor([pos[line[k]] for k in idx])
    rank = (s > s[torch.arange(len(idx)), own][:, None]).sum(1) + 1
    return rank

def show(name, w):
    r = place(w, test_idx)
    out = f"{name}: of {len(test_idx)} kept out, among {len(pos)} lines: first {int((r == 1).sum())} ({100 * float((r == 1).float().mean()):.1f}%), in ten {int((r <= 10).sum())} ({100 * float((r <= 10).float().mean()):.1f}%), median place {int(r.median())}"
    by = {}
    for k, v in zip(test_idx, r.tolist()): by.setdefault(src[k], []).append(v)
    out += " | first, by file: " + ", ".join(f"{s} {sum(x == 1 for x in v)}/{len(v)}" for s, v in sorted(by.items()))
    print(out, flush=True)
    return r

before = show("before", None)
w = torch.nn.Parameter(torch.zeros(Q.shape[1], Q.shape[1]))
opt = torch.optim.AdamW([w], lr=args.lr, weight_decay=args.decay)
ti = torch.tensor(train_idx)
for step in range(1, args.steps + 1):
    idx = ti[torch.randint(len(ti), (args.batch,))]
    q = torch.nn.functional.normalize(Q[idx] + Q[idx] @ w, dim=1); d = torch.nn.functional.normalize(D[idx] + D[idx] @ w, dim=1)
    loss = torch.nn.functional.cross_entropy((q @ d.T) / args.temp, torch.arange(len(idx)))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0 or step == args.steps:
        print(f"step {step}: loss {loss.item():.3f}", flush=True)
after = show("after", w.detach())
won = int(((after == 1) & (before != 1)).sum()); lost = int(((before == 1) & (after != 1)).sum())
print(f"first place among the kept out: won {won}, lost {lost}, bar {2 * math.sqrt(won + lost):.1f}")
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
    Qe = np.stack([qv[s["key"]] for s in searches])
    pool_rank.write_ranks(searches, dict(zip(v["doc_ids"].tolist(), v["docs"])), Qe, Path(args.out) / f"ranks-tune3-{name}.json")
    pool_rank.write_ranks(searches, dict(zip(v["doc_ids"].tolist(), apply(v["docs"]))), apply(Qe), Path(args.out) / f"ranks-pairs-{name}.json")

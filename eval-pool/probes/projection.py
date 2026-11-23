"""A probe: a linear map learnt on top of a model's frozen vectors (the queries' and the lines' alike), from the
judges' grades: does a change of the metric alone, which an index could take without embedding the library again,
carry to searches it did not see?

    python eval-pool/probes/projection.py vec-judged.npz [--kind full|diag|lowrank] [--decay 1e-3] [--steps 300]

Trains on the judged searches that are not the evaluation's (group.json: half, other-half), five folds of them to
choose the weight decay, and reports the evaluation third: the first judged line good, a good one in three.
"""
import argparse, csv, json, math
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("vectors")
ap.add_argument("--kind", default="full")
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--temp", type=float, default=0.05)
args = ap.parse_args()
torch.manual_seed(5786)
z = np.load(args.vectors)
doc = {int(d): torch.tensor(v) for d, v in zip(z["doc_ids"], z["docs"])}
qv = {str(k).split("/")[1]: torch.tensor(v) for k, v in zip(z["keys"], z["queries"])}
group = json.load(open("build/why/group.json"))
grades = {}
for r in csv.DictReader(open("judge/grades.tsv", encoding="utf-8"), delimiter="\t"):
    grades.setdefault(r["qid"], {})[int(r["lineId"])] = int(r["grade"])
dim = next(iter(doc.values())).shape[0]


def lists(qs):
    out = []
    for q in qs:
        ids = [d for d in grades[q] if d in doc]
        g = torch.tensor([grades[q][d] for d in ids], dtype=torch.float32)
        if len(ids) >= 2 and (g >= 2).any() and (g < 2).any():
            out.append((q, qv[q], torch.stack([doc[d] for d in ids]), g))
    return out


class Map(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "diag":
            self.w = torch.nn.Parameter(torch.zeros(dim))
        elif kind == "lowrank":
            self.a = torch.nn.Parameter(torch.zeros(dim, args.rank))
            self.b = torch.nn.Parameter(torch.randn(args.rank, dim) * 0.01)
        else:
            self.w = torch.nn.Parameter(torch.zeros(dim, dim))

    def forward(self, x):
        if self.kind == "diag":
            y = x * (1 + self.w)
        elif self.kind == "lowrank":
            y = x + x @ self.a @ self.b
        else:
            y = x + x @ self.w
        return torch.nn.functional.normalize(y, dim=-1)

    def penalty(self):
        return sum((p ** 2).sum() for n, p in self.named_parameters() if n != "b")


def pack(data):
    """The lists as one padded batch: queries [N, dim], lines [N, L, dim], grades and a mask [N, L]."""
    n, most = len(data), max(len(g) for _, _, _, g in data)
    q = torch.stack([x[1] for x in data])
    d = torch.zeros(n, most, dim); g = torch.zeros(n, most); mask = torch.zeros(n, most, dtype=torch.bool)
    for i, (_, _, lines, grade) in enumerate(data):
        d[i, :len(grade)] = lines; g[i, :len(grade)] = grade; mask[i, :len(grade)] = True
    return q, d, g, mask


def scores(model, q, d):
    if model is None:
        return (d * q[:, None, :]).sum(-1)
    return (model(d) * model(q)[:, None, :]).sum(-1)


def train(data, decay):
    model = Map(args.kind)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    q, d, g, mask = pack(data)
    target = torch.softmax((g * 2.0).masked_fill(~mask, -1e9), dim=1)   # the good lines share the mass, the source more
    for _ in range(args.steps):
        opt.zero_grad()
        s = (scores(model, q, d) / args.temp).masked_fill(~mask, -1e9)
        loss = -(target * torch.log_softmax(s, dim=1)).sum(1).mean()
        (loss + decay * model.penalty()).backward()
        opt.step()
    return model


def measure(model, data):
    q, d, g, mask = pack(data)
    with torch.no_grad():
        s = scores(model, q, d).masked_fill(~mask, -1e9)
    order = torch.gather(g, 1, torch.argsort(-s, dim=1))
    return int((order[:, 0] >= 2).sum()), int((order[:, :3] >= 2).any(1).sum())


train_q = sorted(q for q, x in group.items() if x in ("half", "other-half"))
eval_q = sorted(q for q, x in group.items() if x == "eval")
tr, ev = lists(train_q), lists(eval_q)
print(f"{len(tr)} training searches and {len(ev)} of the evaluation with a good and a not-good judged line; {args.kind}")
print("as the model is: training", measure(None, tr), "evaluation", measure(None, ev))
folds = [tr[i::5] for i in range(5)]
best = None
for decay in (0.3, 0.03, 0.003, 0.0003):
    held = [0, 0]
    for i in range(5):
        model = train([x for j, f in enumerate(folds) if j != i for x in f], decay)
        a, b = measure(model, folds[i])
        held[0] += a; held[1] += b
    print(f"decay {decay}: held-out folds of the training searches: first good {held[0]}, in three {held[1]}", flush=True)
    if best is None or held[0] > best[0]:
        best = (held[0], decay)
model = train(tr, best[1])
print(f"chosen decay {best[1]}: training {measure(model, tr)}, evaluation {measure(model, ev)}")
def firsts(model, data):
    q, d, g, mask = pack(data)
    with torch.no_grad():
        s = scores(model, q, d).masked_fill(~mask, -1e9)
    return (torch.gather(g, 1, torch.argsort(-s, dim=1))[:, 0] >= 2).tolist()


base, new = firsts(None, ev), firsts(model, ev)
a = sum(x and not y for x, y in zip(base, new)); b = sum(y and not x for x, y in zip(base, new))
print(f"evaluation, first good: only as it is {a}, only with the map {b}, noise bar {2 * math.sqrt(a + b):.1f}")
torch.save(model.state_dict(), f"build/why/projection-{args.kind}.pt")

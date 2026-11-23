"""A probe: a linear map on the released model's frozen vectors, learnt from a large reranker's scores of the first
results of real searches nobody judged (Kaggle's poold), and tried on the judged searches, which it never saw: does
what a strong reranker knows carry to other searches through a change of the metric alone?

    python eval-pool/probes/distill_map.py [--kind full|query] [--steps 300]
"""
import argparse, csv, json, math
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="full")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--temp", type=float, default=0.05)
ap.add_argument("--teacher-temp", type=float, default=2.0)
args = ap.parse_args()
torch.manual_seed(5786)

z = np.load("build/why/vec-distill-tune3.npz")
doc = {int(d): torch.tensor(v) for d, v in zip(z["doc_ids"], z["docs"])}
qv = {str(k): torch.tensor(v) for k, v in zip(z["keys"], z["queries"])}
teacher = json.load(open("build/kaggle-output/poold/ranks-tune3-tuned-merge-strip-qwen3-reranker-4b.json"))
dim = 384
rows = []
for key, r in teacher.items():
    ids = [d for d in r["ids"] if d in doc]
    sc = [s for d, s in zip(r["ids"], r["scores"]) if d in doc]
    if len(ids) >= 5 and key in qv:
        rows.append((key, ids, sc))
most = max(len(ids) for _, ids, _ in rows)


def pack(rs):
    n = len(rs)
    q = torch.stack([qv[k] for k, _, _ in rs])
    d = torch.zeros(n, most, dim); t = torch.full((n, most), -1e9); mask = torch.zeros(n, most, dtype=torch.bool)
    for i, (_, ids, sc) in enumerate(rs):
        d[i, :len(ids)] = torch.stack([doc[x] for x in ids]); t[i, :len(ids)] = torch.tensor(sc); mask[i, :len(ids)] = True
    return q, d, t, mask


class Map(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.w = torch.nn.Parameter(torch.zeros(dim, dim))

    def q(self, x):
        return torch.nn.functional.normalize(x + x @ self.w, dim=-1)

    def d(self, x):
        return self.q(x) if self.kind == "full" else x


def scores(model, q, d):
    if model is None:
        return (d * q[:, None, :]).sum(-1)
    return (model.d(d) * model.q(q)[:, None, :]).sum(-1)


def train(rs, decay):
    model = Map(args.kind)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    q, d, t, mask = pack(rs)
    target = torch.softmax(t / args.teacher_temp, dim=1)
    for _ in range(args.steps):
        opt.zero_grad()
        s = (scores(model, q, d) / args.temp).masked_fill(~mask, -1e9)
        loss = -(target * torch.log_softmax(s, dim=1)).sum(1).mean()
        (loss + decay * (model.w ** 2).sum()).backward()
        opt.step()
    return model


def agree(model, rs):
    """In how many searches the student's first is the teacher's first, and the teacher's first's mean place."""
    q, d, t, mask = pack(rs)
    with torch.no_grad():
        s = scores(model, q, d).masked_fill(~mask, -1e9)
    top = t.argmax(1)
    place = (torch.argsort(-s, dim=1) == top[:, None]).float().argmax(1) + 1
    return int((place == 1).sum()), float(place.float().mean())


# the judged searches, never trained on
zj = np.load("build/why/vec-judged-tune3.npz")
jdoc = {int(d): torch.tensor(v) for d, v in zip(zj["doc_ids"], zj["docs"])}
jq = {str(k).split("/")[1]: torch.tensor(v) for k, v in zip(zj["keys"], zj["queries"])}
grades = {}
for r in csv.DictReader(open("judge/grades.tsv", encoding="utf-8"), delimiter="\t"):
    grades.setdefault(r["qid"], {})[int(r["lineId"])] = int(r["grade"])
group = json.load(open("build/why/group.json"))
judged = [q for q in group if any(g >= 2 for g in grades[q].values()) and q in jq]


def judged_first(model):
    out = {}
    with torch.no_grad():
        for q in judged:
            ids = [d for d in grades[q] if d in jdoc]
            D = torch.stack([jdoc[d] for d in ids]); Q = jq[q]
            s = D @ Q if model is None else model.d(D) @ model.q(Q)
            order = [grades[q][ids[i]] for i in torch.argsort(-s).tolist()]
            out[q] = (order[0] >= 2, any(g >= 2 for g in order[:3]), order[0] == 3)
    return out


perm = torch.randperm(len(rows)).tolist()
held = [rows[i] for i in perm[: len(rows) // 5]]
fit = [rows[i] for i in perm[len(rows) // 5:]]
print(f"{len(rows)} searches with the reranker's scores ({len(fit)} to learn from, {len(held)} held); {len(judged)} judged searches to try on; {args.kind}")
print("as the model is: the reranker's first is the model's first in", agree(None, held), "of the held;",
      "judged searches, first good / good in three / the source first:", tuple(sum(v[i] for v in judged_first(None).values()) for i in range(3)))
base = judged_first(None)
best = None
for decay in (0.3, 0.03, 0.003, 0.0003, 0.00003):
    model = train(fit, decay)
    a = agree(model, held)
    j = judged_first(model)
    oa = sum(base[q][0] and not j[q][0] for q in judged); ob = sum(j[q][0] and not base[q][0] for q in judged)
    print(f"decay {decay}: held, the reranker's first is first in {a[0]} (mean place {a[1]:.2f}); fitted {agree(model, fit)[0]} of {len(fit)}; "
          f"judged: {tuple(sum(v[i] for v in j.values()) for i in range(3))}; first good only as it is {oa}, only with the map {ob}, bar {2 * math.sqrt(oa + ob):.1f}", flush=True)
    if best is None or a[0] > best[0]:
        best = (a[0], decay, model)
torch.save(best[2].state_dict(), f"build/why/distill-map-{args.kind}.pt")
print("chosen by the held searches: decay", best[1])

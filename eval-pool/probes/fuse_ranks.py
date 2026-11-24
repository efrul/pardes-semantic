"""A probe: two or more models' ranks files as one, to see whether a second model adds to the first.

    python eval-pool/probes/fuse_ranks.py <out.json> ranks-a.json[@weight] ranks-b.json[@weight] ...

Each model's scores of a search's candidates are standardized (mean 0, deviation 1) and summed, with the weights. Only
the searches and the candidates every file has. The output is a ranks file as pool_rank.py's, for the same measures.
"""
import json
import sys

import numpy as np

out, *specs = sys.argv[1:]
models = []
for spec in specs:
    path, _, weight = spec.partition("@")
    models.append((json.load(open(path, encoding="utf-8")), float(weight or 1)))
fused = {}
for key in models[0][0]:
    if not all(key in ranks for ranks, _ in models):
        continue
    common = set.intersection(*(set(ranks[key]["ids"]) for ranks, _ in models))
    ids = [d for d in models[0][0][key]["ids"] if d in common]
    total = np.zeros(len(ids))
    for ranks, weight in models:
        score = dict(zip(ranks[key]["ids"], ranks[key]["scores"]))
        v = np.array([score[d] for d in ids])
        if len(v):
            total += weight * (v - v.mean()) / (v.std() + 1e-9)
    order = np.argsort(-total, kind="stable")
    fused[key] = {"ids": [ids[k] for k in order], "scores": [round(float(total[k]), 5) for k in order]}
with open(out, "w", encoding="utf-8") as f:
    f.write(json.dumps(fused) + "\n")
print(f"{len(fused)} searches -> {out}")

"""evaluate.py's benchmark by the vectors' first N dimensions, renormalized, of the corpus and the queries alike: what
is left of a model trained with Matryoshka (train_model.py --matryoshka) at the student's size, 384. The dense side
and the app's hybrid (RRF, the dense side weighed 3), hit@10 and nDCG@10 per task; at the full dimension it gives
evaluate.py's numbers.

    python evaluate_dims.py --eval build/eval --vectors build/eval/vectors --model build/model --dims 768,384
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import evaluate as ev
from pardes_semantic import model as m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, required=True, help="embed_corpus.py's vectors of the benchmark's corpus")
    ap.add_argument("--model", type=Path, required=True, help="the int8 ONNX model that embeds the queries")
    ap.add_argument("--dims", default="768,512,384,256", help="comma separated")
    args = ap.parse_args()
    docs = [json.loads(line) for line in (args.eval / "corpus.jsonl").open(encoding="utf-8")]
    queries = [json.loads(line) for line in (args.eval / "queries.jsonl").open(encoding="utf-8")]
    row_of = {d["id"]: i for i, d in enumerate(docs)}
    n, dim = map(int, (args.vectors / "meta.txt").read_text().split())
    vec_ids = np.fromfile(args.vectors / "ids.i64", dtype="<i8")[:n]
    vecs = np.fromfile(args.vectors / "vecs.f32", dtype="<f4")[: n * dim].reshape(n, dim)
    full = np.zeros((len(docs), dim), dtype=np.float32)
    full[[row_of[int(i)] for i in vec_ids]] = vecs
    bm25 = ev.BM25([d["text"] for d in docs])
    embed = ev.QueryEmbedder(args.model, None)
    qvecs = {q["qid"]: embed(q["text"]) for q in queries}
    lexical = {q["qid"]: bm25.search(q["text"], ev.CANDIDATES) for q in queries}
    for d in (int(x) for x in args.dims.split(",")):
        # int8 as the index holds them; its scale is per vector, so cutting before it is renormalizing
        corpus = ev.unit(m.quantize_int8(full[:, :d]))
        per_task = defaultdict(lambda: defaultdict(list))
        for q in queries:
            relevant = {row_of[i] for i in q["relevant"] if i in row_of}
            if not relevant:
                continue
            qv = ev.unit(m.quantize_int8(qvecs[q["qid"]][:d][None, :]))[0]
            sims = corpus @ qv
            dense = np.argpartition(-sims, ev.CANDIDATES)[:ev.CANDIDATES]
            dense = dense[np.argsort(-sims[dense])].tolist()
            hybrid = ev.rrf(lexical[q["qid"]], dense, weights=(1.0, 3.0))
            for name, ranking in (("dense", dense), ("hybrid x3", hybrid)):
                per_task[q["task"]][name].append(ev.metrics(ranking, relevant))
        for task, systems in per_task.items():
            for name, rows in systems.items():
                print(f"d{d} {task:8s} {name:10s} hit@10 {np.mean([r['hit@10'] for r in rows]):.3f} "
                      f"nDCG@10 {np.mean([r['ndcg@10'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()

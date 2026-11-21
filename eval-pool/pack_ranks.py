"""The pool's searches ranked as the app's dense side ranks them with a pack (README.md): the query embedded by the
pack's int8 ONNX model (its prefix, the app's clean, its truncation), each candidate by its own vector in the pack's
vecs.i8, as the index holds it. The candidates the index has no vector for (fewer than three words, short lines merged
into the next) are left out, as ../pool_rank.py --short-context merge leaves them. Beside the same model's ranks by
pool_rank.py (PyTorch on both sides), the same measures show what the pack loses to int8, in its query model and in its
vectors. No GPU: 20 s for the 1,617 searches of pool v13.

    python pack_ranks.py --spec pool.json --vectors vectors-int8 --model model-tuned --out ranks-pack4.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pardes_semantic.clean import clean  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True, help="pool_spec.py's pool.json")
    ap.add_argument("--vectors", type=Path, required=True, help="the pack's vectors: int8's vectors-int8")
    ap.add_argument("--model", type=Path, required=True, help="the pack's query model (pardes-model.json)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    searches = json.loads(args.spec.read_text("utf-8"))["searches"]
    count, dim = (int(x) for x in (args.vectors / "meta.txt").read_text().split())
    ids = np.fromfile(args.vectors / "ids.i64", dtype="<i8")
    vecs = np.memmap(args.vectors / "vecs.i8", dtype=np.int8, mode="r", shape=(count, dim))
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]

    spec = json.loads((args.model / "pardes-model.json").read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(str(args.model / spec["tokenizer"]))
    tok.enable_truncation(spec["queryMaxTokens"])
    sess = ort.InferenceSession(str(args.model / spec["onnx"]), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}

    def embed(text):
        e = tok.encode(spec["queryPrefix"] + clean(text))
        feed = {"input_ids": np.array([e.ids], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask], dtype=np.int64)}
        v = sess.run([spec["output"]], {k: x for k, x in feed.items() if k in names})[0][0]
        return v / np.linalg.norm(v)

    ranks, total, missing = {}, 0, 0
    for s in searches:
        cands = np.array(list(dict.fromkeys(s["candidates"])), dtype=np.int64)
        pos = np.minimum(np.searchsorted(sorted_ids, cands), len(sorted_ids) - 1)
        found = sorted_ids[pos] == cands
        total += len(cands)
        missing += int((~found).sum())
        kept, rows = cands[found], order[pos[found]]
        scores = np.zeros(0, dtype=np.float32)
        if len(kept):
            # Read in the file's order, then put back in the candidates'
            v = vecs[np.sort(rows)].astype(np.float32)[np.argsort(np.argsort(rows))]
            scores = (v / np.linalg.norm(v, axis=1, keepdims=True)) @ embed(s["query"]).astype(np.float32)
        k = np.argsort(-scores, kind="stable")
        ranks[s["key"]] = {"ids": [int(kept[i]) for i in k], "scores": [round(float(scores[i]), 5) for i in k]}
    args.out.write_text(json.dumps(ranks) + "\n", "utf-8")
    print(f"{len(searches)} searches, {total} candidates, {missing} without a vector in the pack -> {args.out}")


if __name__ == "__main__":
    main()

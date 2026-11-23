"""A probe: a multilingual reranker's ONNX file with only the tokens the library can use. Two thirds of the small
reranker's int8 file is its table of 256,000 tokens, and the library's text uses a few thousand of them.

    python eval-pool/probes/cut_vocabulary.py --model build/rerank/small-int8.onnx --tokenizer build/rerank/tokenizer.json \
        --db build/library/seforim.db --out build/rerank/small-int8-cut.onnx [--queries queries.txt] [--sample 1]

Kept: every token a line of the library tokenizes into (cleaned as the index holds it, without points and accents;
--sample N reads one line in N), the tokens of the --queries file's lines, every token made of Hebrew letters, digits
and common punctuation alone (so that any Hebrew text tokenizes into kept tokens), the single ASCII characters and
the tokenizer's added tokens. Any other token becomes <unk>.

The file takes the tokenizer's own ids: a small table at its entry maps each id to its row in the cut table, so the
code that runs the model does not change. Prints the sizes and checks that the scores of a sample of pairs are the
same as the whole file's.
"""
import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from tokenizers import Tokenizer

sys.path.insert(0, ".")
import hub_score  # noqa: E402
from embed_corpus import strip_number  # noqa: E402
from pardes_semantic.clean import clean  # noqa: E402

HEBREW = re.compile(r"^[▁֐-׿0-9 \"'׳״\-.,:;!?()\[\]/]+$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--queries", type=Path, help="a text file, a query a line")
    ap.add_argument("--sample", type=int, default=1, help="one line of the library in this many")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    tok.no_truncation()
    tok.no_padding()
    spec = json.loads(args.tokenizer.read_text("utf-8"))
    vocab = spec["model"]["vocab"]
    keep = {i for t, i in vocab.items() if HEBREW.match(t) or (len(t) == 1 and t.isascii())}
    keep |= {a["id"] for a in spec["added_tokens"]}
    rule = len(keep)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    (top,) = con.execute("SELECT max(id) FROM line").fetchone()
    t0, lines = time.time(), 0
    for lo in range(0, top + 1, 200_000):
        rows = [hub_score.plain(strip_number(clean(c or ""))) for (c,) in con.execute(
            "SELECT content FROM line WHERE id > ? AND id <= ? AND id % ? = 0", (lo, lo + 200_000, args.sample))]
        for e in tok.encode_batch(rows, add_special_tokens=False):
            keep.update(e.ids)
        lines += len(rows)
        if (lo // 200_000) % 5 == 0:
            print(f"  {lines} lines, {len(keep)} tokens, {(time.time() - t0) / 60:.1f} min", flush=True)
    if args.queries:
        for e in tok.encode_batch([hub_score.plain(clean(q)) for q in args.queries.read_text("utf-8").splitlines()],
                                  add_special_tokens=False):
            keep.update(e.ids)
    kept = sorted(keep)
    print(f"{lines} lines of the library; kept {len(kept)} of {len(vocab)} tokens ({rule} by the rule alone)")

    model = onnx.load(str(args.model))
    graph = model.graph
    table = next(i for i in graph.initializer if len(i.dims) == 2 and i.dims[0] == len(vocab))
    gather = next(n for n in graph.node if n.op_type == "Gather" and n.input[0] == table.name)
    ids_name = gather.input[1]
    rows = numpy_helper.to_array(table)
    unk = vocab[spec["model"]["unk_token"]]
    remap = np.full(len(vocab), kept.index(unk), dtype=np.int32)
    remap[kept] = np.arange(len(kept), dtype=np.int32)
    cut = numpy_helper.from_array(np.ascontiguousarray(rows[kept]), table.name)
    graph.initializer.remove(table)
    graph.initializer.extend([cut, numpy_helper.from_array(remap, "token_rows")])
    gather.input[1] = "token_row_ids"
    at = list(graph.node).index(gather)
    graph.node.insert(at, helper.make_node("Gather", ["token_rows", ids_name], ["token_row_ids"], name="token_rows_gather"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.out))
    print(f"{args.model.stat().st_size / 1e6:.0f} MB -> {args.out.stat().st_size / 1e6:.0f} MB: {args.out}")

    # the same scores, on lines of the library against each other
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    whole = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    small = ort.InferenceSession(str(args.out), options, providers=["CPUExecutionProvider"])
    sample = [hub_score.plain(strip_number(clean(c or ""))) for (c,) in con.execute(
        "SELECT content FROM line WHERE id % 97003 = 5 LIMIT 60")]
    worst = 0.0
    for a, b in zip(sample, sample[1:]):
        ids = np.array([tok.encode(" ".join(a.split()[:6]), b).ids[:256]], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        worst = max(worst, abs(float(whole.run(None, feed)[0][0, 0]) - float(small.run(None, feed)[0][0, 0])))
    print(f"{len(sample) - 1} pairs: the largest difference between the two files' scores is {worst:.6f}")


if __name__ == "__main__":
    main()

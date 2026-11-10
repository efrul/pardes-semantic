"""The library's vectors against their lines (BENCHMARK.md, section 4): N random rows of vecs.i8 against their line's
text embedded again by the int8 ONNX model (cosine), and against the next row's line (what a shifted id would give).
Vectors whose embed-info.json says shortMerge (embed_corpus.py --short-merge, merge_vectors.py) are checked against the
text they were embedded from, the short lines just before the line and the line, and none of their lines may be short;
with stripNumber too, without a number in parentheses at a line's start.

    python check_vectors.py build/vectors-int8 build/model-student seforim.db [N]
"""

import json
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_corpus import strip_number  # noqa: E402
from pardes_semantic.clean import clean  # noqa: E402


def main() -> None:
    vec_dir, model_dir, db = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    spec = json.loads((model_dir / "pardes-model.json").read_text(encoding="utf-8"))
    count, dim = (int(x) for x in (vec_dir / "meta.txt").read_text().split())
    ids = np.fromfile(vec_dir / "ids.i64", dtype="<i8")
    vecs = np.memmap(vec_dir / "vecs.i8", dtype=np.int8, mode="r", shape=(count, dim))
    assert len(ids) == count
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tok = Tokenizer.from_file(str(model_dir / spec["tokenizer"]))
    tok.enable_truncation(spec["documentMaxTokens"])
    sess = ort.InferenceSession(str(model_dir / spec["onnx"]), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    info_path = vec_dir / "embed-info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    merge, lead_max = info.get("shortMerge") or 0, info.get("mergeLead") or 3
    prep = (lambda t: strip_number(clean(t))) if info.get("stripNumber") else clean

    def text_of(line_id):
        """The text the line's vector was embedded from: as embed_corpus.py --short-merge reads the library."""
        book, index, content = con.execute("SELECT bookId, lineIndex, content FROM line WHERE id = ?",
                                           (line_id,)).fetchone()
        text = prep(content)
        if not merge:
            return text
        lead = []
        for k in range(1, lead_max + 1):
            row = con.execute("SELECT content FROM line WHERE bookId = ? AND lineIndex = ?", (book, index - k)).fetchone()
            t = prep(row[0] or "") if row else ""
            if not t or len(t.split()) > merge:
                break
            lead.insert(0, t)
        return " ".join(lead + [text])

    def embed(text):
        e = tok.encode(spec["documentPrefix"] + text)
        feed = {"input_ids": np.array([e.ids], dtype=np.int64), "attention_mask": np.array([e.attention_mask], dtype=np.int64)}
        out = sess.run([spec["output"]], {k: v for k, v in feed.items() if k in names})[0][0]
        return out / np.linalg.norm(out)

    rng = random.Random(5812)
    same, shifted, short = [], [], 0
    for r in rng.sample(range(count - 1), n):
        text = text_of(int(ids[r]))
        own = prep(con.execute("SELECT content FROM line WHERE id = ?", (int(ids[r]),)).fetchone()[0])
        short += merge > 0 and len(own.split()) <= merge
        v = embed(text)
        a, b = vecs[r].astype(np.float32), vecs[r + 1].astype(np.float32)
        same.append(float(a @ v / np.linalg.norm(a)))
        shifted.append(float(b @ v / np.linalg.norm(b)))
    print(f"{n} rows: cosine to their line min {min(same):.3f} mean {np.mean(same):.3f}; "
          f"to the next row's line mean {np.mean(shifted):.3f}"
          + (f"; short lines merged ({merge} words or fewer): {short} of the rows are short lines" if merge else ""))


if __name__ == "__main__":
    main()

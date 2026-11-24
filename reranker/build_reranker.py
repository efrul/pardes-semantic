"""The reranker as the app loads it, the folder `pardes-reranker/`: a ready-made multilingual cross-encoder of the Hub
(Apache-2.0), its weights in int8 and its vocabulary cut to the tokens the library can use (README.md).

    python reranker/build_reranker.py --out build/pardes-reranker                 # with reranker/tokens.txt
    python reranker/build_reranker.py --out build/pardes-reranker --db seforim.db [--queries queries.txt] --write-tokens

The folder: `pardes-reranker.onnx`, `tokenizer.json`, `pardes-reranker.json` (what the app reads: the files, their
checksums, how many of a page's first lines it orders again and from how long a query), the source's own `LICENSE`
and `NOTICE`, which its licence asks every copy to carry, and this repository's `NOTICE.md`, which says what was
changed.

- **The model:** SOURCE at REVISION, its own ONNX export in float32, quantized here as the app's embedding model is
  (dynamic int8 with reduce_range: full-range int8 saturates on an x86 CPU without VNNI).
- **The tokens** (tokens.txt, ids of the source's tokenizer): every token a line of the library tokenizes into
  (cleaned as the index holds it, without its number, points and accents) and the --queries file's lines; every token
  made of Hebrew letters, digits and common punctuation alone; the tokenizer's byte tokens and special tokens; and
  every token that is a part of one of these, so that the merges that built them are all there. With --db the list is
  made again (--write-tokens saves it) and the cut tokenizer is checked against the whole one on the library's lines.
- **The tokenizer** is the source's with the kept tokens alone, numbered again in their order, and the merges among
  them. On a text whose tokens are all kept it gives what the whole tokenizer gives; a character it does not know
  becomes its bytes. The model's table of tokens keeps the same rows, so the ids are the rows.
- **Checked:** the cut file scores a sample of pairs as the whole int8 file does, exactly, and int8 is compared with
  float32 on them.
"""

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import hub_score  # noqa: E402
from embed_corpus import strip_number  # noqa: E402
from pardes_semantic.clean import CLEANER_ID, clean  # noqa: E402
from pardes_semantic.samples import PAIRS  # noqa: E402

SOURCE = "Horizon-Labs/multilingual-reranker-small"
REVISION = "23139fec6f1f448cdf89a5ac2d2c6bf5e8c7b398"
LICENSE = "Apache-2.0"
ONNX_NAME = "pardes-reranker.onnx"
SPEC_NAME = "pardes-reranker.json"
FORMAT = 1
# Measured on the judged searches (eval-pool/results-why.md): the page's first lines the reranker reads, the fewest
# words of a query it reads them for, a pair's tokens at most, and by how much its first line must pass the page's own
# first to take its place
LINES, MIN_WORDS, MAX_TOKENS, QUERY_MAX_TOKENS, MARGIN = 6, 3, 256, 64, 1.5
SPECIAL = ("<pad>", "<eos>", "<bos>", "<unk>", "<mask>")
HEBREW = re.compile(r"^[▁֐-׿0-9 \"'׳״\-.,:;!?()\[\]/]+$")
BYTE = re.compile(r"^<0x[0-9A-F]{2}>$")


def line_text(content: str) -> str:
    """A line as the reranker reads it: as the index holds it, without its number, points and accents."""
    return hub_score.plain(strip_number(clean(content or "")))


def query_text(query: str) -> str:
    return hub_score.plain(clean(query))


def library_tokens(tok, db: Path, queries: Path | None) -> set[int]:
    """The ids of the source's tokens that the library's lines and the queries tokenize into."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    (top,) = con.execute("SELECT max(id) FROM line").fetchone()
    seen, t0, lines = set(), time.time(), 0
    for lo in range(0, top + 1, 200_000):
        rows = [line_text(c) for (c,) in con.execute("SELECT content FROM line WHERE id > ? AND id <= ?", (lo, lo + 200_000))]
        for e in tok.encode_batch(rows, add_special_tokens=False):
            seen.update(e.ids)
        lines += len(rows)
        if (lo // 200_000) % 5 == 0:
            print(f"  {lines} lines, {len(seen)} tokens, {(time.time() - t0) / 60:.1f} min", flush=True)
    if queries:
        for e in tok.encode_batch([query_text(q) for q in queries.read_text("utf-8").splitlines()], add_special_tokens=False):
            seen.update(e.ids)
    print(f"{lines} lines of the library: {len(seen)} tokens")
    return seen


def closed(seen: set[int], vocab: dict[str, int]) -> list[int]:
    """[seen] with the Hebrew, byte and special tokens, and every token that is a part of one of them."""
    keep = set(seen)
    keep |= {i for t, i in vocab.items() if HEBREW.match(t) or BYTE.match(t) or (len(t) == 1 and t.isascii())}
    keep |= {vocab[t] for t in SPECIAL}
    text = {i: t for t, i in vocab.items()}
    for i in list(keep):
        s = text[i]
        for a in range(len(s)):
            for b in range(a + 1, len(s) + 1):
                j = vocab.get(s[a:b])
                if j is not None:
                    keep.add(j)
    return sorted(keep)


def cut_tokenizer(spec: dict, kept: list[int]) -> dict:
    """The source's tokenizer.json with the [kept] tokens alone, numbered again in their order."""
    vocab = spec["model"]["vocab"]
    row = {old: new for new, old in enumerate(kept)}
    out = json.loads(json.dumps(spec))
    out["truncation"], out["padding"] = None, None
    out["model"]["vocab"] = {t: row[i] for t, i in sorted(vocab.items(), key=lambda x: x[1]) if i in row}
    out["model"]["merges"] = [m for m in spec["model"]["merges"]
                              if all(vocab.get(t) in row for t in (m[0], m[1], m[0] + m[1]))]
    # The special tokens alone are matched in a text before it is tokenized; the source's other added tokens (runs of
    # line breaks and tabs, unused ones) are ordinary tokens here
    out["added_tokens"] = [dict(a, id=row[a["id"]]) for a in spec["added_tokens"] if a["content"] in SPECIAL]
    for name, token in out["post_processor"]["special_tokens"].items():
        token["ids"] = [row[vocab[name]]]
    return out


def write_tokens(path: Path, kept: list[int]) -> None:
    """The kept ids as ranges, a range a line."""
    lines, start, last = [], kept[0], kept[0]
    for i in kept[1:] + [None]:
        if i is not None and i == last + 1:
            last = i
            continue
        lines.append(str(start) if start == last else f"{start}-{last}")
        if i is not None:
            start = last = i
    path.write_text("\n".join(lines) + "\n", "utf-8")


def read_tokens(path: Path) -> list[int]:
    kept = []
    for line in path.read_text("utf-8").split():
        a, _, b = line.partition("-")
        kept.extend(range(int(a), int(b or a) + 1))
    return kept


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scores(session, tok, pairs: list[tuple[str, str]]) -> list[float]:
    tok.enable_truncation(max_length=MAX_TOKENS, strategy="only_second")
    out = []
    for q, d in pairs:
        ids = np.array([tok.encode(q, d).ids], dtype=np.int64)
        out.append(float(session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0, 0]))
    tok.no_truncation()  # a text alone is tokenized after this, which truncating the second of a pair refuses
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, default=HERE / "tokens.txt")
    ap.add_argument("--db", type=Path, help="the library: the tokens list is made again from its lines")
    ap.add_argument("--queries", type=Path, help="with --db: a text file, a query a line, whose tokens are kept too")
    ap.add_argument("--write-tokens", action="store_true", help="with --db: save the list made as --tokens")
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from onnx import numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from tokenizers import Tokenizer

    source_onnx = Path(hf_hub_download(SOURCE, "onnx/model.onnx", revision=REVISION))
    source_tokenizer = Path(hf_hub_download(SOURCE, "tokenizer.json", revision=REVISION))
    notices = {name: Path(hf_hub_download(SOURCE, name, revision=REVISION)) for name in ("LICENSE", "NOTICE")}
    spec = json.loads(source_tokenizer.read_text("utf-8"))
    vocab = spec["model"]["vocab"]
    whole = Tokenizer.from_file(str(source_tokenizer))
    whole.no_truncation()
    whole.no_padding()

    if args.db:
        kept = closed(library_tokens(whole, args.db, args.queries), vocab)
        if args.write_tokens:
            write_tokens(args.tokens, kept)
            print(f"-> {args.tokens}")
    else:
        kept = read_tokens(args.tokens)
    assert kept == sorted(set(kept)) and all(vocab[t] in set(kept) for t in SPECIAL)
    print(f"{len(kept)} of {len(vocab)} tokens kept")

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.out / "tokenizer.json"
    tokenizer_path.write_text(json.dumps(cut_tokenizer(spec, kept), ensure_ascii=False), "utf-8")
    cut = Tokenizer.from_file(str(tokenizer_path))

    with tempfile.TemporaryDirectory() as tmp:
        int8 = Path(tmp) / "int8.onnx"
        quantize_dynamic(str(source_onnx), str(int8), weight_type=QuantType.QInt8, reduce_range=True,
                         op_types_to_quantize=["MatMul", "Gather"])
        model = onnx.load(str(int8))
        table = next(i for i in model.graph.initializer if len(i.dims) == 2 and i.dims[0] == len(vocab))
        rows = numpy_helper.to_array(table)
        model.graph.initializer.remove(table)
        model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(rows[kept]), table.name))
        onnx.save(model, str(args.out / ONNX_NAME))

        # The cut file against the whole int8 file, and int8 against float32, on the repository's sample pairs
        options = ort.SessionOptions()
        pairs = [(query_text(q), line_text(p)) for q, p in PAIRS]
        ours = scores(ort.InferenceSession(str(args.out / ONNX_NAME), options, providers=["CPUExecutionProvider"]), cut, pairs)
        full = scores(ort.InferenceSession(str(int8), options, providers=["CPUExecutionProvider"]), whole, pairs)
        fp32 = scores(ort.InferenceSession(str(source_onnx), options, providers=["CPUExecutionProvider"]), whole, pairs)
    same = max(abs(a - b) for a, b in zip(ours, full))
    drift = [abs(a - b) for a, b in zip(ours, fp32)]
    order = sum((a > b) == (c > d) for (a, c), (b, d) in zip(zip(ours, fp32), zip(ours[1:], fp32[1:])))
    print(f"{len(pairs)} pairs: the cut file against the whole int8 file, largest difference {same:.6f}; int8 against "
          f"float32, mean {sum(drift) / len(drift):.3f}, largest {max(drift):.3f}; neighbours in the same order "
          f"{order} of {len(pairs) - 1}")
    if same > 1e-4:
        raise SystemExit("the cut file does not score as the whole one: a token of the samples is not kept")

    if args.db:
        # The cut tokenizer against the whole one, on lines of the library
        con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        texts = [line_text(c) for (c,) in con.execute("SELECT content FROM line WHERE id % 23 = 7")]
        row = {old: new for new, old in enumerate(kept)}
        bad = 0
        for i in range(0, len(texts), 20_000):
            chunk = texts[i:i + 20_000]
            for a, b in zip(whole.encode_batch(chunk, add_special_tokens=False), cut.encode_batch(chunk, add_special_tokens=False)):
                bad += [row.get(x) for x in a.ids] != b.ids
        print(f"{len(texts)} lines of the library: the cut tokenizer differs from the whole one on {bad}")
        if bad:
            raise SystemExit("the cut tokenizer does not tokenize the library as the whole one does")

    onnx_sha = sha256(args.out / ONNX_NAME)
    manifest = {
        "format": FORMAT,
        "id": f"reranker-{onnx_sha[:7]}",
        "onnx": ONNX_NAME,
        "tokenizer": "tokenizer.json",
        "source": SOURCE,
        "revision": REVISION,
        "license": LICENSE,
        "cleaner": CLEANER_ID,
        "lines": LINES,
        "minWords": MIN_WORDS,
        "maxTokens": MAX_TOKENS,
        "queryMaxTokens": QUERY_MAX_TOKENS,
        "margin": MARGIN,
        "tokens": len(kept),
        "onnxSha256": onnx_sha,
        "tokenizerSha256": sha256(tokenizer_path),
        "attribution": f"{SOURCE} by Horizon Labs (Apache-2.0), a fine-tuned mmBERT-small (MIT); quantized to int8 and "
                       "its vocabulary cut by the Pardes project. See LICENSE, NOTICE and NOTICE.md.",
    }
    (args.out / SPEC_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8")
    for name, path in notices.items():
        shutil.copyfile(path, args.out / name)
    shutil.copyfile(HERE.parent / "NOTICE.md", args.out / "NOTICE.md")
    for f in sorted(args.out.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
    print(f"{manifest['id']} -> {args.out}")


if __name__ == "__main__":
    main()

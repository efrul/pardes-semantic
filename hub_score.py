"""Texts through a model of the Hub that is not ours, to measure another family beside our models (pool_rank.py --st,
pool_rerank.py): an embedder that sentence-transformers loads, or a reranker that reads the query and the line
together. On every GPU there is, a process each: a model of billions of parameters fits a T4 once, and is not copied
between processes.

    python hub_score.py embed  --model Qwen/Qwen3-Embedding-4B --texts texts.json --out vectors.npy [--prompt query]
    python hub_score.py rerank --model Qwen/Qwen3-Reranker-4B  --pairs pairs.json --out scores.json

texts.json is a list of texts, pairs.json a list of [query, text]; the vectors are normalized, in the texts' order.
--prompt: the model's own prompt of that name before each text (its "query" prompt). A reranker whose name holds
"Qwen3-Reranker" is asked as its card asks it, yes or no, and scored by the two logits; a path to an .onnx file is a
cross-encoder's export, run by ONNX Runtime on the CPU with the tokenizer.json beside it (the file an app would load,
quantized as it would be); any other is a sentence-transformers CrossEncoder.

Other families' tokenizers keep the Hebrew points and accents, which ours strips (pardes_semantic/clean.py): plain()
takes them out first, so that a line is the same text to every model.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_MARKS = re.compile("[֑-ֽֿ-ׇ]")
QWEN_TASK = "Given a search query, retrieve relevant passages that answer the query"
QWEN_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
               "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def plain(text: str) -> str:
    """[text] without the Hebrew points and accents, the maqaf a space."""
    return " ".join(_MARKS.sub("", text).replace("־", " ").split())


def _device_and_dtype(dtype: str):
    import torch
    if torch.cuda.is_available():
        return "cuda", getattr(torch, dtype)
    torch.set_num_threads(os.cpu_count() or 1)
    return "cpu", torch.float32


def embed_here(model: str, texts: list[str], prompt: str | None, dtype: str, batch: int, max_tokens: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    device, torch_dtype = _device_and_dtype(dtype)
    st = SentenceTransformer(model, device=device, model_kwargs={"torch_dtype": torch_dtype})
    st.max_seq_length = max_tokens
    if prompt and prompt not in (st.prompts or {}):
        print(f"{model} has no '{prompt}' prompt: the texts as they are", flush=True)
        prompt = None
    order = sorted(range(len(texts)), key=lambda k: len(texts[k]))
    out, t0 = [None] * len(texts), time.time()
    for i in range(0, len(order), batch):
        chunk = order[i:i + batch]
        vectors = st.encode([texts[k] for k in chunk], batch_size=len(chunk), normalize_embeddings=True,
                            prompt_name=prompt, show_progress_bar=False)
        for k, v in zip(chunk, vectors):
            out[k] = v
        if (i // batch) % 200 == 0:
            print(f"  {i}/{len(texts)} texts, {(time.time() - t0) / 60:.1f} min", flush=True)
    return np.asarray(out, dtype=np.float32)


def rerank_here(model: str, pairs: list, dtype: str, batch: int, max_tokens: int) -> list[float]:
    import torch
    device, torch_dtype = _device_and_dtype(dtype)
    order = sorted(range(len(pairs)), key=lambda k: len(pairs[k][1]))
    if "qwen3-reranker" in model.lower():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype).to(device).eval()
        yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        pre = tok.encode(QWEN_PREFIX, add_special_tokens=False)
        suf = tok.encode(QWEN_SUFFIX, add_special_tokens=False)

        @torch.inference_mode()
        def score(chunk):
            rows = [f"<Instruct>: {QWEN_TASK}\n<Query>: {q}\n<Document>: {d}" for q, d in chunk]
            ids = tok(rows, add_special_tokens=False, truncation=True,
                      max_length=max_tokens - len(pre) - len(suf))["input_ids"]
            enc = tok.pad({"input_ids": [pre + x + suf for x in ids]}, padding=True, return_tensors="pt").to(device)
            logits = lm(**enc).logits[:, -1, :].float()
            return (logits[:, yes] - logits[:, no]).tolist()
    elif model.endswith(".onnx"):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(model).with_name("tokenizer.json")))
        tok.enable_truncation(max_length=max_tokens, strategy="only_second")
        pad = tok.token_to_id("<pad>") or 0
        options = ort.SessionOptions()
        options.intra_op_num_threads = os.cpu_count() or 1
        session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
        typed = "token_type_ids" in {i.name for i in session.get_inputs()}

        def score(chunk):
            rows = [tok.encode(q, d).ids for q, d in chunk]
            width = max(len(r) for r in rows)
            ids = np.full((len(rows), width), pad, dtype=np.int64)
            mask = np.zeros((len(rows), width), dtype=np.int64)
            for k, r in enumerate(rows):
                ids[k, :len(r)], mask[k, :len(r)] = r, 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if typed:
                feed["token_type_ids"] = np.zeros_like(ids)
            return session.run(None, feed)[0].reshape(-1).tolist()
    else:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(model, max_length=max_tokens, device=device)

        def score(chunk):
            return ce.predict([tuple(p) for p in chunk], batch_size=len(chunk), show_progress_bar=False).tolist()
    out, t0 = [0.0] * len(pairs), time.time()
    for i in range(0, len(order), batch):
        chunk = order[i:i + batch]
        for k, s in zip(chunk, score([pairs[k] for k in chunk])):
            out[k] = float(s)
        if (i // batch) % 200 == 0:
            print(f"  {i}/{len(pairs)} pairs, {(time.time() - t0) / 60:.1f} min", flush=True)
    return out


def _shards(what: str, model: str, items: list, extra: list[str], load) -> list:
    """[items] through `hub_score.py what` on every GPU, a process each on its share; in this process when there is
    one GPU or none, or the model is a file for the CPU."""
    import torch
    gpus = torch.cuda.device_count()
    if gpus <= 1 or model.endswith(".onnx"):
        return None
    from huggingface_hub import snapshot_download
    snapshot_download(model)  # once, before the processes that would each fetch it
    key = "--texts" if what == "embed" else "--pairs"
    with tempfile.TemporaryDirectory() as tmp:
        runs = []
        for g in range(gpus):
            src, out = Path(tmp) / f"in-{g}.json", Path(tmp) / f"out-{g}"
            src.write_text(json.dumps(items[g::gpus], ensure_ascii=False), "utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            cmd = [sys.executable, str(Path(__file__).resolve()), what, "--model", model, key, str(src), "--out",
                   str(out), *extra]
            runs.append((subprocess.Popen(cmd, env=env), out))
        parts = []
        for process, out in runs:
            if process.wait() != 0:
                raise SystemExit(f"hub_score.py {what} failed on a GPU")
            parts.append(load(out))
    merged = [None] * len(items)
    for g, part in enumerate(parts):
        merged[g::gpus] = list(part)
    return merged


def embed(model: str, texts: list[str], prompt: str | None = None, dtype: str = "float16", batch: int = 8,
          max_tokens: int = 512) -> np.ndarray:
    extra = ["--dtype", dtype, "--batch", str(batch), "--max-tokens", str(max_tokens)] + (["--prompt", prompt] if prompt else [])
    merged = _shards("embed", model, texts, extra, lambda p: np.load(str(p) + ".npy"))
    if merged is None:
        return embed_here(model, texts, prompt, dtype, batch, max_tokens)
    return np.asarray(merged, dtype=np.float32)


def rerank(model: str, pairs: list, dtype: str = "float16", batch: int = 8, max_tokens: int = 512) -> list[float]:
    extra = ["--dtype", dtype, "--batch", str(batch), "--max-tokens", str(max_tokens)]
    merged = _shards("rerank", model, pairs, extra, lambda p: json.loads(Path(p).read_text("utf-8")))
    if merged is None:
        return rerank_here(model, pairs, dtype, batch, max_tokens)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("embed", "rerank"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--texts", type=Path)
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    if args.what == "embed":
        texts = json.loads(args.texts.read_text("utf-8"))
        np.save(str(args.out) + ".npy", embed_here(args.model, texts, args.prompt, args.dtype, args.batch, args.max_tokens))
    else:
        pairs = json.loads(args.pairs.read_text("utf-8"))
        args.out.write_text(json.dumps(rerank_here(args.model, pairs, args.dtype, args.batch, args.max_tokens)), "utf-8")


if __name__ == "__main__":
    main()

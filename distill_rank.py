"""Distills a teacher into the small student as distill_model.py does (its layers, a PCA projection to --dim), learning
besides the teacher's vectors (MSE) the teacher's order: for each query of a batch of pairs, the teacher's distribution
over the batch's passages (KL), as the search goes by the order of the scores, not by the vectors themselves. A student
that learnt only the vectors kept about half of what the teacher gained, and lost most on short queries
(eval-pool/results-run3.md).

    python distill_rank.py --teacher build/teacher --pairs build/train/pairs.jsonl --links 120000 \
        --more-pairs build/train/chavruta.jsonl --queries build/train/queries.txt --out build/student
    python distill_rank.py --teacher build/teacher --pairs pairs.jsonl --links 40 --dim 32 --limit 200 \
        --max-steps 3 --device cpu --out /tmp/smoke                                                       # a smoke test

The batches alternate: a batch of pairs (the vectors of its queries and passages, and the KL of their scores) and a
batch of texts (the vectors only: the searches of --queries and the texts of the pairs again). The vectors are learnt by
their direction, 2(1 - cosine) to the projected teacher's, and not by distill_model.py's MSE: the app normalizes them,
and the teacher's are not normalized (mean pooling only), so that their MSE (183 at the start) would leave the KL
(about 0.5) no weight. Writes a sentence-transformers directory,
as distill_model.py, that export_onnx.py --source and embed_corpus.py --source take.
"""

import argparse
import copy
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from distill_model import kept_layers
from pardes_semantic import model as m
from train_model import load_st, precision


def read_pairs(path: Path, limit: int = 0) -> list[tuple[str, str]]:
    out = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        out.append((r["anchor"], r["positive"]))
        if limit and len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", help="a local model (train_model.py); default: the Dicta model")
    ap.add_argument("--pairs", type=Path, required=True, help="train's pairs.jsonl: the links' pairs")
    ap.add_argument("--links", type=int, default=120_000, help="the first N of --pairs")
    ap.add_argument("--more-pairs", type=Path, action="append", default=[],
                    help="more pairs, all of them (build_chavruta_pairs.py's, train-v2/build_pairs.py's); repeatable")
    ap.add_argument("--queries", type=Path, help="searches distilled on too, one per line (build_distill_queries.py)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=7)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16,
                    help="pairs a GPU in a batch of pairs (its texts: twice as many), all GPUs at once (DataParallel): "
                         "32 texts a GPU, as distill_model.py's; 64 pairs on one 16 GB T4 ran out of memory")
    ap.add_argument("--text-batches", choices=("searches", "all"), default="searches",
                    help="the texts of the batches of texts: the searches only (the pairs' texts are learnt in the "
                         "batches of pairs), or every text once more")
    ap.add_argument("--teacher-embeddings", type=Path,
                    help="the teacher's embeddings of these same texts, from an earlier run (its teacher-embeddings.npy): "
                         "not computed again")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.05, help="of both score matrices (the model's scale of 20)")
    ap.add_argument("--rank-weight", type=float, default=1.0, help="the KL beside the MSE")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=5785)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (a smoke test)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N texts of --queries (a smoke test)")
    ap.add_argument("--deadline", type=float, default=0, help="Unix time to stop by, saving the model (0: none)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    gpu = args.device.startswith("cuda")

    pairs = read_pairs(args.pairs, args.links)
    for path in args.more_pairs:
        more = read_pairs(path)
        print(f"{len(more)} pairs of {path.name}")
        pairs += more
    pairs = list(dict.fromkeys(pairs))
    searches = []
    if args.queries:
        searches = [q for q in args.queries.read_text("utf-8").splitlines() if q.strip()]
        if args.limit:
            searches = searches[:args.limit]
    # Every text once: the queries with the query prefix, the passages with the document one, as the app embeds them
    index: dict[str, int] = {}

    def text_id(text: str) -> int:
        return index.setdefault(text, len(index))

    pair_ids = [(text_id(m.QUERY_PREFIX + a), text_id(m.DOCUMENT_PREFIX + p)) for a, p in pairs]
    for q in searches:
        text_id(m.QUERY_PREFIX + q)
    texts = list(index)
    print(f"{len(pairs)} pairs, {len(searches)} searches: {len(texts)} texts")
    if len(texts) <= args.dim:
        raise SystemExit(f"{len(texts)} texts cannot give a {args.dim}-dimension PCA")

    cache = args.out / "teacher-embeddings.npy"
    given = args.teacher_embeddings
    if given and np.load(given, mmap_mode="r").shape[0] == len(texts):
        # The same inputs give the same texts in the same order: the earlier run's embeddings are these texts'
        teacher = load_st(args.teacher, "cpu")
        raw = np.load(given).astype(np.float32)
        print(f"Teacher embeddings from {given}")
    elif given:
        raise SystemExit(f"{given} holds {np.load(given, mmap_mode='r').shape[0]} embeddings, not {len(texts)}: "
                         "not these texts")
    else:
        teacher = load_st(args.teacher, args.device)
    if given:
        pass
    elif cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(texts):
        raw = np.load(cache)
        print(f"Teacher embeddings from {cache}")
    else:
        dtype = torch.bfloat16 if precision(gpu)["bf16"] else torch.float16
        with torch.autocast("cuda", dtype=dtype, enabled=gpu):
            raw = teacher.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=True)
        raw = raw.astype(np.float32)
        args.out.mkdir(parents=True, exist_ok=True)
        np.save(cache, raw)

    # The PCA of the teacher's embeddings, as distill_model.py: the student's Dense starts as it, the targets are in it
    mean = raw.mean(axis=0)
    _, _, vt = np.linalg.svd(raw - mean, full_matrices=False)
    components = vt[: args.dim]
    explained = float(((raw - mean) @ components.T).var(axis=0).sum() / (raw - mean).var(axis=0).sum())
    print(f"PCA to {args.dim} dimensions keeps {explained:.1%} of the variance")
    targets = torch.from_numpy((raw @ components.T).astype(np.float32))
    del raw
    gc.collect()

    teacher.to("cpu")
    if gpu:
        torch.cuda.empty_cache()
    student = copy.deepcopy(teacher)
    auto = student[0].auto_model
    total = len(auto.transformer_encoder)
    keep = kept_layers(total, args.layers)
    auto.transformer_encoder = torch.nn.ModuleList([auto.transformer_encoder[i] for i in keep])
    auto.config.num_hidden_layers = len(keep)
    print(f"Student: layers {keep} of {total}")
    from sentence_transformers.models import Dense
    dense = Dense(auto.config.hidden_size, args.dim, bias=False, activation_function=torch.nn.Identity())
    with torch.no_grad():
        dense.linear.weight.copy_(torch.from_numpy(components))
    student.append(dense)
    del teacher
    gc.collect()
    student.to(args.device)
    student.train()
    gpus = torch.cuda.device_count() if gpu else 0
    # Every GPU at once: a batch of pairs is split between them, its scores computed on the whole batch
    forward = torch.nn.DataParallel(student) if gpus > 1 else student
    batch = args.batch * max(1, gpus)
    print(f"{max(1, gpus)} {'GPUs' if gpus > 1 else 'device'}: {batch} pairs a batch")

    # The batches, in a shuffled order: those of pairs, and those of texts (every text once more)
    pair_order = list(range(len(pair_ids)))
    rng.shuffle(pair_order)
    text_order = list(range(len(texts))) if args.text_batches == "all" else \
        sorted({index[m.QUERY_PREFIX + q] for q in searches})
    rng.shuffle(text_order)
    batches = [("pairs", pair_order[i:i + batch]) for i in range(0, len(pair_order), batch)]
    batches += [("texts", text_order[i:i + 2 * batch]) for i in range(0, len(text_order), 2 * batch)]
    rng.shuffle(batches)
    steps = len(batches) if not args.max_steps else min(args.max_steps, len(batches))
    print(f"{steps} steps: {sum(k == 'pairs' for k, _ in batches)} of pairs, {sum(k == 'texts' for k, _ in batches)} "
          f"of texts")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, fused=gpu)
    warmup = max(1, int(steps * 0.05))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup))))
    prec = precision(gpu)
    amp_dtype = torch.bfloat16 if prec["bf16"] else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=gpu and prec["fp16"])

    def direction(out: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
        return (F.normalize(out, dim=-1) - F.normalize(want, dim=-1)).pow(2).sum(-1).mean()

    def embed(ids: list[int]) -> torch.Tensor:
        features = student.tokenize([texts[i] for i in ids])
        features = {k: v.to(args.device) for k, v in features.items()}
        with torch.autocast("cuda", dtype=amp_dtype, enabled=gpu):
            return forward(features)["sentence_embedding"].float()

    t0 = time.time()
    log = {"mse": torch.zeros((), device=args.device), "kl": torch.zeros((), device=args.device), "n": 0}
    for step, (kind, chunk) in enumerate(batches[:steps]):
        if args.deadline and time.time() > args.deadline:
            print(f"Deadline reached at step {step}: stopping and saving", flush=True)
            break
        if kind == "pairs":
            q_ids = [pair_ids[k][0] for k in chunk]
            p_ids = [pair_ids[k][1] for k in chunk]
            out = embed(q_ids + p_ids)
            want = targets[q_ids + p_ids].to(args.device)
            mse = direction(out, want)
            qs, ps = F.normalize(out[:len(chunk)], dim=-1), F.normalize(out[len(chunk):], dim=-1)
            qt, pt = F.normalize(want[:len(chunk)], dim=-1), F.normalize(want[len(chunk):], dim=-1)
            student_scores = qs @ ps.T / args.temperature
            teacher_scores = qt @ pt.T / args.temperature
            # Both ways, as the search ranks passages for a query and the teacher's scores hold for either side
            kl = 0.5 * (F.kl_div(F.log_softmax(student_scores, -1), F.softmax(teacher_scores, -1), reduction="batchmean")
                        + F.kl_div(F.log_softmax(student_scores.T, -1), F.softmax(teacher_scores.T, -1),
                                   reduction="batchmean"))
            loss = mse + args.rank_weight * kl
            log["kl"] += kl.detach()
        else:
            out = embed(chunk)
            mse = direction(out, targets[chunk].to(args.device))
            loss = mse
        log["mse"] += mse.detach()
        log["n"] += 1
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        schedule.step()
        if (step + 1) % 200 == 0 or step + 1 == steps:
            n = max(1, log["n"])
            print(f"step {step + 1}/{steps}: direction {log['mse'].item() / n:.5f}, kl {log['kl'].item() / n:.4f}, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            log = {"mse": torch.zeros((), device=args.device), "kl": torch.zeros((), device=args.device), "n": 0}

    student.eval()
    student.save(str(args.out))
    cache.unlink(missing_ok=True)
    (args.out / "distill-info.json").write_text(json.dumps(
        {"teacher": args.teacher or m.MODEL_ID, "layers": keep, "of": total, "dim": args.dim, "pcaExplained": explained,
         "texts": len(texts), "pairs": len(pairs), "searches": len(searches), "method": "direction+kl",
         "temperature": args.temperature, "rankWeight": args.rank_weight, "batch": batch,
         "textBatches": args.text_batches, "steps": step + 1, "ofSteps": len(batches)}, indent=2) + "\n", "utf-8")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

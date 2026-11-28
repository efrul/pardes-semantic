"""Fine-tunes the Dicta model on the library's links (build_train_pairs.py): stage C of the plan.

    python train_model.py --pairs build/train/pairs.jsonl --out build/teacher            # on a GPU (~2-4 h on one A100)
    python train_model.py --pairs build/train/pairs.jsonl --out /tmp/smoke --max-steps 2 --batch-size 4 --device cpu
    python train_model.py --source build/teacher --pairs build/train/pairs.jsonl --limit 20000 \
        --questions build/train/questions.jsonl --question-repeat 10 --out build/tuned      # the teacher on the questions

An interrupted run (a Colab session that ends) starts again from its last checkpoint when run again with the same
--out: keep --out on persistent storage (Google Drive).

The loss is the one Dicta trained the model with (CachedMultipleNegativesRankingLoss): each query against its passage
and every other passage of the batch, so a large batch (--batch-size, default 256) gives many negatives at the memory
of a small one (--mini-batch). The queries get the model's "query: " prefix, as the app's do; no text twice in a batch.

--questions adds the training questions (train-v2/build_pairs.py), each --question-repeat times, to the pairs: they
are few, and a reader's way of asking is what the links' pairs lack. It is repeatable: the feedback's searches
(feedback/collect.py's pairs.jsonl) are questions too. --more adds other pairs as they are, such as
build_chavruta_pairs.py's (Hebrew explanation -> the Gemara line).

--negatives uses the pairs' hard negatives ("negatives": train-v2's graded ones, the lines the feedback's users graded
0), in the same batches as the other pairs: a batch of questions only could not be filled, since the two questions of
a target share its text. Every row gets one negative: its own in turn, one more each time its query comes again (a
question repeated, a search's other good lines), and for a row with none, a passage of the links' pairs past --limit,
which no row learns (one more random negative, as the batch's other passages are).

--matryoshka trains the vector's first dimensions too, each a vector of its own (MatryoshkaLoss over the same loss):
TRAINING_PLAN.md's teacher v2 (768,512,384,256), whose first 384 can be measured against the 384 of the student.

--device xla trains on a TPU (PyTorch/XLA: Kaggle's TPU v5e-8, one chip; tpu_probe.py, RUN5_PLAN.md, 3): every text
padded to the same length, the forward in bf16, no checkpoints (a run takes minutes there).

Writes a sentence-transformers directory that export_onnx.py --source and embed_corpus.py --source take.
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

from pardes_semantic import model as m

TRAIN_MAX_TOKENS = 256


def negatives_column(rows: list[dict], spare: list[str], seed: int = 0) -> list[str]:
    """One negative for each of [rows]: its hard negatives in turn, the next one each time its anchor comes again, or a
    passage of [spare] (the positives of the rows when there is none) for a row without, never its own positive."""
    rng = random.Random(seed)
    pool = spare or [r["positive"] for r in rows]
    turns = Counter()
    column = []
    for r in rows:
        hard = [n for n in r.get("negatives") or [] if n != r["positive"]]
        if hard:
            column.append(hard[turns[r["anchor"]] % len(hard)])
            turns[r["anchor"]] += 1
            continue
        negative = pool[rng.randrange(len(pool))]
        for _ in range(10):
            if negative != r["positive"]:
                break
            negative = pool[rng.randrange(len(pool))]
        column.append(negative)
    return column


def precision(gpu: bool) -> dict[str, bool]:
    """bf16 on Ampere and later (A100, L4...), fp16 on older GPUs (Colab's free T4 has no bf16), none on a CPU."""
    if not gpu:
        return {"bf16": False, "fp16": False}
    import torch

    bf16 = torch.cuda.is_bf16_supported()
    return {"bf16": bf16, "fp16": not bf16}


def last_checkpoint(out: Path) -> str | None:
    """The newest checkpoint of an interrupted run in out/checkpoints, to resume from (a Colab session ends)."""
    ckpts = sorted((out / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(ckpts[-1]) if ckpts else None


def deadline_callback(deadline: float, save: bool = True):
    """Stops the training cleanly (and so saves the model) at [deadline] (Unix seconds): a Kaggle run that outlives its
    session loses its output, a run that stops a little early keeps a usable model. [save]: a checkpoint too (not on a
    TPU, where the trainer saves only the weights)."""
    import time

    from transformers import TrainerCallback

    class Deadline(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if time.time() > deadline:
                print(f"Deadline reached at step {state.global_step}: stopping and saving")
                control.should_training_stop = True
                control.should_save = save
            return control

    return Deadline()


def load_st(source: str | None, device: str):
    from sentence_transformers import SentenceTransformer

    name, revision = (m.MODEL_ID, m.MODEL_REVISION) if source is None else (source, None)
    st = SentenceTransformer(name, revision=revision, trust_remote_code=True, device=device,
                             prompts={"query": m.QUERY_PREFIX, "document": m.DOCUMENT_PREFIX})
    st.max_seq_length = TRAIN_MAX_TOKENS
    return st


def for_xla(st) -> None:
    """Readies [st] for a TPU. XLA compiles a graph for each shape it meets, so every text is padded to
    TRAIN_MAX_TOKENS (a shape for each length would compile again at nearly every step). The forward runs in bf16, its
    embeddings back in fp32 for the loss. The cached loss copies the random state of CUDA's devices before its first
    pass, to repeat the model's dropout in the second; there is none to copy on a TPU, so a model with dropout is
    refused (Dicta's NeoBERT and our students have none)."""
    import functools
    import importlib

    import torch

    # The module, not the class of the same name that sentence_transformers.losses holds
    cached = importlib.import_module("sentence_transformers.losses.CachedMultipleNegativesRankingLoss")

    if any(isinstance(mod, torch.nn.Dropout) and mod.p > 0 for mod in st.modules()):
        raise SystemExit("--device xla: the model has dropout, which the cached loss could not repeat on a TPU")
    first = st[0]
    first.tokenize = functools.partial(first.tokenize, padding="max_length")
    forward = st.forward

    def bf16_forward(features, *args, **kwargs):
        with torch.autocast("xla", dtype=torch.bfloat16):
            out = forward(features, *args, **kwargs)
        out["sentence_embedding"] = out["sentence_embedding"].float()
        return out

    st.forward = bf16_forward

    class NoRandomState:
        def __init__(self, *tensors) -> None:
            pass

        def __enter__(self) -> None:
            pass

        def __exit__(self, *exc) -> None:
            pass

    cached.RandContext = NoRandomState


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", help="start from this local model instead of the Dicta one")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--mini-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N steps (a smoke test)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N pairs")
    ap.add_argument("--questions", type=Path, action="append", default=[],
                    help="train-v2/build_pairs.py's questions, added to the pairs; repeatable (feedback/collect.py's)")
    ap.add_argument("--question-repeat", type=int, default=1, help="each question this many times")
    ap.add_argument("--more", type=Path, action="append", default=[],
                    help="more pairs in the same jsonl (build_chavruta_pairs.py), added as they are; repeatable")
    ap.add_argument("--negatives", action="store_true",
                    help="train on the pairs' hard negatives too (their \"negatives\"), one a row")
    ap.add_argument("--matryoshka", help="the dimensions the loss trains as vectors of their own, e.g. 768,512,384,256")
    ap.add_argument("--save-steps", type=int, default=200, help="checkpoint every N steps (resumed automatically)")
    ap.add_argument("--deadline", type=float, default=0, help="Unix time to stop by, saving the model (0: none)")
    ap.add_argument("--keep-checkpoints", type=int, default=2,
                    help="checkpoints kept (with the optimizer: ~4.4 GB each for the full-size model)")
    args = ap.parse_args()
    xla = args.device == "xla"
    if xla:
        os.environ.setdefault("PJRT_DEVICE", "TPU")

    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    rows = [json.loads(line) for line in args.pairs.open(encoding="utf-8")]
    # The links' passages that no row learns: the negatives of the rows without hard ones
    spare = [r["positive"] for r in rows[args.limit:]] if args.limit else []
    if args.limit:
        rows = rows[:args.limit]
    links = len(rows)
    for path in args.questions:
        questions = [json.loads(line) for line in path.open(encoding="utf-8")]
        rows += questions * args.question_repeat
        print(f"{len(questions)} questions of {path.name}, {args.question_repeat} times each")
    for path in args.more:
        more = [json.loads(line) for line in path.open(encoding="utf-8")]
        rows += more
        print(f"{len(more)} pairs of {path.name}")
    columns = {"anchor": [r["anchor"] for r in rows], "positive": [r["positive"] for r in rows]}
    prompts = {"anchor": m.QUERY_PREFIX, "positive": m.DOCUMENT_PREFIX}
    if args.negatives:
        columns["negative"] = negatives_column(rows, spare)
        prompts["negative"] = m.DOCUMENT_PREFIX
        hard = sum(1 for r in rows if any(n != r["positive"] for n in r.get("negatives") or []))
        print(f"{hard} rows with a hard negative, {len(rows) - hard} with a random one")
    data = Dataset.from_dict(columns)
    print(f"{len(data)} pairs ({links} of links)")

    # On a TPU the trainer moves the model to the chip
    st = load_st(args.source, "cpu" if xla else args.device)
    if xla:
        for_xla(st)
    loss = CachedMultipleNegativesRankingLoss(st, mini_batch_size=args.mini_batch)
    if args.matryoshka:
        from sentence_transformers.losses import MatryoshkaLoss

        loss = MatryoshkaLoss(st, loss, matryoshka_dims=[int(d) for d in args.matryoshka.split(",")])
    gpu = args.device.startswith("cuda")
    training = SentenceTransformerTrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        **precision(gpu),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts=prompts,
        logging_steps=20,
        save_strategy="no" if xla else "steps",
        save_steps=args.save_steps,
        save_total_limit=args.keep_checkpoints,
        report_to="none",
        use_cpu=not (gpu or xla),
        dataloader_drop_last=True,
        # The fused AdamW runs on CUDA and CPU only: on a TPU the same AdamW, unfused
        **({"optim": "adamw_torch"} if xla else {}),
    )
    callbacks = [deadline_callback(args.deadline, save=not xla)] if args.deadline else None
    trainer = SentenceTransformerTrainer(model=st, args=training, train_dataset=data, loss=loss, callbacks=callbacks)
    resume = last_checkpoint(args.out)
    if resume:
        print(f"Resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    if xla:
        st.to("cpu")
    st.save(str(args.out))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

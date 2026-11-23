"""The first candidates of each search, ordered again by a reranker of the Hub that reads the query and the line
together (eval-pool/README.md): what a second model over the first results would give.

    python pool_rerank.py --db seforim.db --spec pool.json --ranks ranks-tuned.json \
        --model Qwen/Qwen3-Reranker-4B --top 20 --out ranks-tuned-rr.json

Each search's first --top candidates in --ranks' order (pool_rank.py's ranks file of the same spec) are scored by the
reranker (hub_score.py) and put in its order; the candidates after them stay behind, in the order they had. Without
--ranks, the first --top of the spec's own candidates: for a spec whose candidates are a search's judged lines. A line
is read as the index holds it, cleaned and without its number, and without the Hebrew points and accents.
ranks.json as pool_rank.py's; the scores of the reranked lines are the reranker's, the others' below them.
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import hub_score
from embed_corpus import MIN_WORDS, strip_number
from pardes_semantic.clean import clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--ranks", type=Path, help="the order whose first candidates are reranked; the spec's when absent")
    ap.add_argument("--model", required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--only", help="comma separated: only the searches whose key starts with one of these")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    searches = json.loads(args.spec.read_text("utf-8"))["searches"]
    if args.only:
        searches = [s for s in searches if s["key"].startswith(tuple(args.only.split(",")))]
    base = json.loads(args.ranks.read_text("utf-8")) if args.ranks else None
    order = {s["key"]: (base[s["key"]]["ids"] if base else list(dict.fromkeys(s["candidates"]))) for s in searches}
    ids = sorted({d for o in order.values() for d in o[:args.top]})
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    texts = {}
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for line_id, content in con.execute(f"SELECT id, content FROM line WHERE id IN ({','.join('?' * len(chunk))})",
                                            chunk):
            text = hub_score.plain(strip_number(clean(content or "")))
            if len(text.split()) >= MIN_WORDS:
                texts[line_id] = text
    pairs, where = [], []
    for s in searches:
        query = hub_score.plain(clean(s["query"]))
        for d in order[s["key"]][:args.top]:
            if d in texts:
                pairs.append([query, texts[d]])
                where.append((s["key"], d))
    print(f"{len(searches)} searches, {len(pairs)} pairs for {args.model}", flush=True)
    t0 = time.time()
    scores = dict(zip(where, hub_score.rerank(args.model, pairs, args.dtype, args.batch, args.max_tokens)))
    print(f"scored in {(time.time() - t0) / 60:.1f} min", flush=True)

    out = {}
    for s in searches:
        key = s["key"]
        head = sorted((d for d in order[key][:args.top] if (key, d) in scores), key=lambda d: -scores[(key, d)])
        rest = [d for d in order[key] if (key, d) not in scores]
        floor = min((scores[(key, d)] for d in head), default=0.0) - 1
        out[key] = {"ids": head + rest,
                    "scores": [round(scores[(key, d)], 5) for d in head] + [round(floor - k * 1e-3, 5) for k in range(len(rest))]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n", "utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

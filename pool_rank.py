"""Ranks each search's candidates by a model, to compare models on the evaluations without embedding the library
(eval-pool/README.md).

    python pool_rank.py --db seforim.db --spec pool.json --out ranks-teacher.json [--source build/teacher]

pool.json (eval-pool/pool_spec.py): the searches, each with its query and its candidates, the app's lexical and dense
ones. Every candidate the semantic index would hold (MIN_WORDS words or more, cleaned as embed_corpus.py cleans) is
embedded with the model as a document, the query as a query, and the candidates ranked by their cosine. Without
--source, the Dicta model. ranks.json: {key: {"ids": [...], "scores": [...]}}, the best first.
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from embed_corpus import MIN_WORDS
from pardes_semantic import model as m
from pardes_semantic.clean import clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", help="a local model (train_model.py, distill_model.py), not the Dicta one")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch-tokens", type=int, default=8192, help="padded tokens per batch")
    ap.add_argument("--doc-source", help="embed the candidates with this model instead (a teacher), mapped into "
                    "--source's space: --source then embeds only the queries")
    ap.add_argument("--fit", type=int, default=20_000, help="--doc-source: candidates the linear map is fitted on")
    ap.add_argument("--proj-out", type=Path, help="--doc-source: also the ranking by the doc model's own queries, "
                    "mapped alike (what the map keeps of the doc model)")
    ap.add_argument("--short-context", choices=("none", "heading", "neighbors", "merge", "drop"), default="none",
                    help="how a short line (--short-words or fewer) is embedded: as it is, after its table of contents "
                    "entry, between its neighbours' words, not at all but before the next line's text, or not at all")
    ap.add_argument("--short-words", type=int, default=8)
    args = ap.parse_args()
    import torch

    searches = json.loads(args.spec.read_text("utf-8"))["searches"]
    ids = sorted({d for s in searches for d in s["candidates"]})
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    texts = {}
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for line_id, content in con.execute(f"SELECT id, content FROM line WHERE id IN ({','.join('?' * len(chunk))})",
                                            chunk):
            text = clean(content)
            if len(text.split()) >= MIN_WORDS:
                texts[line_id] = m.DOCUMENT_PREFIX + text
    print(f"{len(searches)} searches, {len(ids)} candidates, {len(texts)} of them in the index", flush=True)
    if args.short_context != "none":
        texts = short_context(con, texts, args.short_context, args.short_words)

    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    queries = [m.QUERY_PREFIX + s["query"] for s in searches]
    if args.doc_source:
        # Asymmetric: the library's lines by a teacher (embedded once, offline), the queries by the small model the app
        # runs. The student learnt the teacher's vectors projected to its dimensions, so a linear map, fitted by least
        # squares on candidates both embed, takes the teacher's vectors into the student's space.
        teacher = m.load(args.device, dtype, args.doc_source)
        tvec = embed_docs(teacher, texts, args.batch_tokens)
        tq = embed_queries(teacher, queries)
        del teacher
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        loaded = m.load(args.device, dtype, args.source)
        rng = np.random.default_rng(5785)
        ids = sorted(tvec)
        sample = [ids[k] for k in rng.permutation(len(ids))[:args.fit]]
        svec = embed_docs(loaded, {d: texts[d] for d in sample}, args.batch_tokens)
        cut = int(len(sample) * 0.9)
        T = np.stack([tvec[d] for d in sample]).astype(np.float64)
        S = np.stack([svec[d] for d in sample]).astype(np.float64)
        w = np.linalg.lstsq(T[:cut], S[:cut], rcond=None)[0]
        held = normalize(T[cut:] @ w) * normalize(S[cut:])
        print(f"map {T.shape[1]} -> {S.shape[1]} fitted on {cut} candidates: cosine to the student on the other "
              f"{len(sample) - cut}, mean {held.sum(1).mean():.4f}, min {held.sum(1).min():.4f}", flush=True)
        w = np.linalg.lstsq(T, S, rcond=None)[0]
        vectors = dict(zip(ids, normalize(np.stack([tvec[d] for d in ids]) @ w).astype(np.float32)))
        qv = normalize(embed_queries(loaded, queries)).astype(np.float32)
        if args.proj_out:
            write_ranks(searches, vectors, normalize(tq @ w).astype(np.float32), args.proj_out)
    else:
        loaded = m.load(args.device, dtype, args.source)
        vectors = embed_docs(loaded, texts, args.batch_tokens)
        qv = embed_queries(loaded, queries)
    print(f"embedded in {(time.time() - t0) / 60:.1f} min", flush=True)
    write_ranks(searches, vectors, qv, args.out)


NEIGHBOUR_WORDS = 40  # of each neighbour, so that the short line stays the middle of its text


def short_context(con, texts: dict, mode: str, short_words: int) -> dict:
    """[texts] with their short lines embedded otherwise (docs/SEARCH_ORDER_ISSUES.md, 1): the semantic side brings
    headings and fragments of a line ("מאי טעמא? אמר רב יהודה:") first, as a vector of a few words is near many queries."""
    def rows(where_ids):
        out = {}
        ids = sorted(where_ids)
        for i in range(0, len(ids), 900):
            chunk = ids[i:i + 900]
            out |= {r[0]: r[1:] for r in con.execute(
                f"SELECT id, bookId, lineIndex, tocEntryId, content FROM line WHERE id IN ({','.join('?' * len(chunk))})",
                chunk)}
        return out

    def at(places):
        """{(book, index): cleaned text} of the lines at [places]."""
        by_book = {}
        for b, x in places:
            by_book.setdefault(b, set()).add(x)
        out = {}
        for b, xs in by_book.items():
            xs = sorted(xs)
            for i in range(0, len(xs), 900):
                chunk = xs[i:i + 900]
                for x, c in con.execute(f"SELECT lineIndex, content FROM line WHERE bookId = ? AND lineIndex IN "
                                        f"({','.join('?' * len(chunk))})", [b, *chunk]):
                    out[(b, x)] = clean(c or "")
        return out

    info = rows(texts)
    words = {d: len(clean(info[d][3] or "").split()) for d in texts}
    short = {d for d in texts if words[d] <= short_words}
    out = dict(texts)
    if mode == "drop":
        for d in short:
            del out[d]
    elif mode == "heading":
        # line_toc maps every line to its entries (line.tocEntryId only the lines the table of contents points to):
        # the deepest of them, the siman's or the seif's rather than the book part's
        entry = {}
        shorts = sorted(short)
        for i in range(0, len(shorts), 900):
            chunk = shorts[i:i + 900]
            for line_id, e, level in con.execute(
                    f"SELECT lt.lineId, lt.tocEntryId, te.level FROM line_toc lt JOIN tocEntry te ON te.id = lt.tocEntryId "
                    f"WHERE lt.lineId IN ({','.join('?' * len(chunk))})", chunk):
                if line_id not in entry or level > entry[line_id][1]:
                    entry[line_id] = (e, level)
        entries = {e for e, _ in entry.values()}
        heads = {}
        for i in range(0, len(sorted(entries)), 900):
            chunk = sorted(entries)[i:i + 900]
            heads |= {e: (parent, text) for e, parent, text in con.execute(
                f"SELECT te.id, te.parentId, tt.text FROM tocEntry te JOIN tocText tt ON tt.id = te.textId "
                f"WHERE te.id IN ({','.join('?' * len(chunk))})", chunk)}
        parents = {p for p, _ in heads.values() if p}
        ptext = {}
        for i in range(0, len(sorted(parents)), 900):
            chunk = sorted(parents)[i:i + 900]
            ptext |= {e: t for e, t in con.execute(f"SELECT te.id, tt.text FROM tocEntry te JOIN tocText tt ON "
                                                     f"tt.id = te.textId WHERE te.id IN ({','.join('?' * len(chunk))})",
                                                     chunk)}
        for d in short:
            e = entry.get(d, (None, 0))[0]
            if e in heads:
                parent, text = heads[e]
                head = clean(" ".join(t for t in (ptext.get(parent), text) if t))
                if head and head not in texts[d]:
                    out[d] = m.DOCUMENT_PREFIX + head + " " + texts[d][len(m.DOCUMENT_PREFIX):]
    elif mode == "neighbors":
        near = at({(info[d][0], info[d][1] + k) for d in short for k in (-1, 1)})
        for d in short:
            b, x = info[d][0], info[d][1]
            before = near.get((b, x - 1), "").split()[-NEIGHBOUR_WORDS:]
            after = near.get((b, x + 1), "").split()[:NEIGHBOUR_WORDS]
            out[d] = m.DOCUMENT_PREFIX + " ".join(before + [texts[d][len(m.DOCUMENT_PREFIX):]] + after)
    elif mode == "merge":
        for d in short:
            del out[d]
        prev = at({(info[d][0], info[d][1] - k) for d in out for k in (1, 2, 3)})
        for d in list(out):
            b, x = info[d][0], info[d][1]
            lead = []
            for k in (1, 2, 3):  # the short lines just before it: a heading, then a fragment
                t = prev.get((b, x - k))
                if t is None or not t or len(t.split()) > short_words:
                    break
                lead.insert(0, t)
            if lead:
                out[d] = m.DOCUMENT_PREFIX + " ".join(lead) + " " + out[d][len(m.DOCUMENT_PREFIX):]
    changed = sum(out.get(d) != texts[d] for d in texts)
    print(f"short lines ({short_words} words or fewer): {len(short)} of {len(texts)}; {mode}: {changed} texts "
          f"changed or left out", flush=True)
    return out


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def embed_docs(loaded, texts: dict, batch_tokens: int) -> dict:
    """{line id: vector} of [texts], as embed_corpus.py: by length, as many as fit in the token budget once padded."""
    docs = sorted(texts)
    lengths = {}
    for i in range(0, len(docs), 20_000):
        enc = loaded.tokenizer([texts[d] for d in docs[i:i + 20_000]], truncation=True, max_length=m.DOCUMENT_MAX_TOKENS)
        lengths |= {d: len(x) for d, x in zip(docs[i:i + 20_000], enc["input_ids"])}
    docs.sort(key=lambda d: (lengths[d], d))
    vectors = {}
    i = 0
    while i < len(docs):
        j = i + 1
        while j < len(docs) and (j - i + 1) * lengths[docs[j]] <= batch_tokens:
            j += 1
        batch = docs[i:j]
        vectors |= dict(zip(batch, m.embed(loaded, [texts[d] for d in batch], m.DOCUMENT_MAX_TOKENS)))
        i = j
    return vectors


def embed_queries(loaded, queries: list[str]) -> np.ndarray:
    return np.concatenate([m.embed(loaded, queries[k:k + 64], m.QUERY_MAX_TOKENS) for k in range(0, len(queries), 64)])


def write_ranks(searches: list, vectors: dict, qv: np.ndarray, path: Path) -> None:
    out = {}
    for s, q in zip(searches, qv):
        cands = [d for d in dict.fromkeys(s["candidates"]) if d in vectors]
        scores = np.stack([vectors[d] for d in cands]) @ q if cands else np.zeros(0)
        order = np.argsort(-scores, kind="stable")
        out[s["key"]] = {"ids": [cands[k] for k in order], "scores": [round(float(scores[k]), 5) for k in order]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + "\n", "utf-8")
    print(f"-> {path}")


if __name__ == "__main__":
    main()

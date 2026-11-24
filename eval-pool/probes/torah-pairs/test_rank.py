"""A model on the questions build_torah_pairs.py kept out: each question's own line among the kept lines and other
lines of the same books.

    python eval-pool/probes/torah-pairs/test_rank.py <model dir> <name> [--others 20000]
    python eval-pool/probes/torah-pairs/test_rank.py --compare <name> <name>

The candidates are chosen once (build/train/torah-pairs-test-cands.json) and are the same for every model: the kept
lines, and lines drawn from their books, none within three lines of a kept line (a neighbour may answer too). Writes
build/train/torah-pairs-test-<name>.npz, the place of each question's line; --compare pairs two models question by
question, against twice the noise.
"""
import argparse, collections, json, math, random, sqlite3, sys, time
from pathlib import Path

import numpy as np

TEST = Path("build/train/torah-pairs-test.jsonl")
CANDS = Path("build/train/torah-pairs-test-cands.json")


def places(name):
    z = np.load(f"build/train/torah-pairs-test-{name}.npz", allow_pickle=True)
    return z["place"], [str(t) for t in z["type"]]


def report(name, place, kinds):
    def line(label, v):
        v = np.sort(np.asarray(v)); n = len(v)
        return (f"{label}: {n} questions | first {int((v == 1).sum())} ({100 * (v == 1).mean():.1f}%) | in ten "
                f"{int((v <= 10).sum())} ({100 * (v <= 10).mean():.1f}%) | in 100 {int((v <= 100).sum())} | median place {int(v[n // 2])}")
    print(line(name, place))
    for kind in sorted(set(kinds)):
        print("  " + line(kind, [p for p, k in zip(place, kinds) if k == kind]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--others", type=int, default=20000)
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        a, b = args.compare
        pa, kinds = places(a); pb, _ = places(b)
        report(a, pa, kinds); report(b, pb, kinds)
        for label, hit in (("first", lambda p: p == 1), ("in ten", lambda p: p <= 10)):
            x = int((hit(pa) & ~hit(pb)).sum()); y = int((hit(pb) & ~hit(pa)).sum())
            bar = 2 * math.sqrt(x + y)
            verdict = "within the noise" if abs(x - y) <= bar else f"{a if x > y else b} better"
            print(f"{label}: only {a} {x}, only {b} {y}, bar {bar:.1f}: {verdict}")
        return
    import torch
    sys.path[:0] = ["."]
    import pool_rank
    from embed_corpus import MIN_WORDS, strip_number
    from pardes_semantic import model as m
    from pardes_semantic.clean import clean
    torch.set_num_threads(4)
    test = [json.loads(line) for line in TEST.open(encoding="utf-8")]
    con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
    if CANDS.exists():
        ids = json.loads(CANDS.read_text())
    else:
        rng = random.Random(5811)
        kept = [t["line"] for t in test]
        place = {}
        for i in range(0, len(kept), 900):
            chunk = kept[i:i + 900]
            for lid, book, index in con.execute(f"SELECT id, bookId, lineIndex FROM line WHERE id IN ({','.join('?' * len(chunk))})", chunk):
                place[lid] = (book, index)
        near = {(b, x + d) for b, x in place.values() for d in range(-3, 4)}
        weight = collections.Counter(b for b, _ in place.values())
        others = set()
        for book, n in weight.items():
            rows = con.execute("SELECT id, lineIndex FROM line WHERE bookId = ? AND heRef IS NOT NULL", (book,)).fetchall()
            rows = [lid for lid, x in rows if (book, x) not in near]
            others.update(rng.sample(rows, min(len(rows), round(args.others * n / len(kept)) + 1)))
        ids = sorted(set(kept) | others)
        CANDS.write_text(json.dumps(ids))
    texts = {}
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for lid, content in con.execute(f"SELECT id, content FROM line WHERE id IN ({','.join('?' * len(chunk))})", chunk):
            t = strip_number(clean(content or ""))
            if len(t.split()) >= MIN_WORDS:
                texts[lid] = m.DOCUMENT_PREFIX + t
    loaded = m.load("cpu", torch.float32, args.model)
    t0 = time.time()
    vec = pool_rank.embed_docs(loaded, texts, 8192)
    print(len(vec), "lines embedded in", round((time.time() - t0) / 60, 1), "min", flush=True)
    docs = sorted(vec); M = np.stack([vec[d] for d in docs]); pos = {d: k for k, d in enumerate(docs)}
    test = [t for t in test if t["line"] in pos]
    Q = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + t["anchor"] for t in test])
    place = np.array([int((M @ q > M[pos[t["line"]]] @ q).sum()) + 1 for t, q in zip(test, Q)])
    kinds = [t["type"] for t in test]
    np.savez(f"build/train/torah-pairs-test-{args.name}.npz", place=place, type=np.array(kinds), line=np.array([t["line"] for t in test]))
    report(f"{args.name}, among {len(docs)} lines", place, kinds)


if __name__ == "__main__":
    main()

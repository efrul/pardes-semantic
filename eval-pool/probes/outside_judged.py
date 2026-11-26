"""A probe: real searches judged outside this repository by a strong language model, as a test of our models. The
companion package of the private collection (collection-notes.md, in the private working dataset) holds 150 searches
users sent from the apps, each with the passages several engines found over the whole library, every passage judged
relevant or not by Claude Opus 5.5. None of them is among the searches judge/ graded.

    python eval-pool/probes/outside_judged.py spec --rounds <rounds.jsonl> --labels <the judge's export .jsonl> \
        --db seforim.db --out build/outside/spec.json --judged build/outside/judged.json
    python pool_rank.py --db seforim.db --spec build/outside/spec.json --source build/kaggle-output/tune3/tuned \
        --device cpu --dtype float32 --short-context merge --strip-number --out build/outside/ranks-tune3.json
    python eval-pool/probes/outside_judged.py score --judged build/outside/judged.json \
        --ranks tune3=build/outside/ranks-tune3.json --ranks tune14=build/outside/ranks-tune14.json

spec: each judged passage becomes a line of ours, by its reference in its book (the reference written as ours is, and
the text checked: most of its words in our line), else by its place in the book. Left out: a passage only Zayit's
engine found (this repository uses nothing of Zayit's semantic search, CLAUDE.md), and a passage not in our library.
The candidates of a search are its judged passages only, so this ranks the judged pool: it says which model puts the
relevant passages above the others, not what a model finds that no engine did.

score: in each search with a relevant and a not relevant line among the lines both models rank, whether the first is
relevant, the reciprocal rank of the first relevant one in the first ten, and nDCG@10; two models paired, against
twice the noise (2 sqrt(b + c) of the searches where only one of them is right). --without-pardes leaves out the
lines only Pardes's own engines found, since the pool holds what the pack's model already puts high.
Prints numbers only.
"""
import argparse
import json
import math
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path[:0] = ["."]
from pardes_semantic.clean import clean  # noqa: E402

ZAYIT = 8 | 16                      # the engine bits of Zayit's semantic search (zayit_q, zayit_p)
PARDES = 32 | 64 | 512 | 1024       # Pardes's model, alone or with a head over it
MARKS = re.compile(r"[֑-ׇ]")
OVERLAP = 0.6                       # the share of the passage's words our line must hold


def words(text: str) -> list[str]:
    return re.sub(r"[^א-ת ]", " ", MARKS.sub("", clean(text or "").replace("־", " "))).split()


def ref_key(ref: str) -> str:
    return " ".join(ref.replace(",", " ").split())


def labels_of(path: Path) -> dict:
    import ast
    out = {}
    for raw in path.open(encoding="utf-8"):
        row = json.loads(raw)
        state = row["state"]
        if isinstance(state, str):
            state = ast.literal_eval(state)      # the export writes a Python dict's repr: literals only, nothing is run
        out[row["qid"]] = state.get("labels", {})
    return out


def spec(args) -> None:
    rows = [json.loads(raw) for raw in args.rounds.open(encoding="utf-8")]
    rounds = [r for r in rows if "q" in r]
    labels = labels_of(args.labels)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    book_of = {title: bid for bid, title in con.execute("SELECT id, title FROM book")}
    lines_of = {}

    def book_lines(bid):
        if bid not in lines_of:
            by_ref, by_index = defaultdict(list), {}
            for lid, index, ref, content in con.execute(
                    "SELECT id, lineIndex, heRef, content FROM line WHERE bookId = ?", (bid,)):
                by_index[index] = (lid, content)
                if ref:
                    by_ref[ref_key(ref)].append((lid, content))
            lines_of[bid] = (by_ref, by_index)
        return lines_of[bid]

    def overlap(text, content):
        mine = words(text)
        return sum(w in set(words(content)) for w in mine) / len(mine) if mine else 0.0

    why = Counter()
    searches, judged = [], {}
    for r in rounds:
        q = r["q"]
        marks = labels.get(q["id"])
        if not marks:
            why["a search not judged"] += 1
            continue
        bits = {c[0]: c[1] for c in q["cands"]}
        cands, grades = [], {}
        for pid, mark in marks.items():
            why["passages judged"] += 1
            if bits.get(pid, 0) and not bits[pid] & ~ZAYIT:
                why["left out: only Zayit's engine found it"] += 1
                continue
            p = r["passages"].get(pid)
            ref = (p or {}).get("r") or ""
            title = ref.split(",")[0].strip()
            bid = book_of.get(title)
            if p is None or bid is None:
                why["left out: not in our library"] += 1
                continue
            by_ref, by_index = book_lines(bid)
            found = None
            for lid, content in by_ref.get(ref_key(title + " " + ref[len(title):].lstrip(" ,")), []):
                if overlap(p["t"], content) >= OVERLAP:
                    found = lid
                    break
            if found is None:
                for index in (p.get("l"), (p.get("l") or 0) - 1, (p.get("l") or 0) + 1):
                    hit = by_index.get(index)
                    if hit and overlap(p["t"], hit[1]) >= OVERLAP:
                        found = hit[0]
                        why["found by its place, not its reference"] += 1
                        break
            if found is None:
                why["left out: no line of ours holds it"] += 1
                continue
            why["mapped to a line of ours"] += 1
            relevant = 1 if mark == 1 else 0
            grades[found] = max(grades.get(found, 0), relevant)
            if found not in cands:
                cands.append(found)
            if bits.get(pid, 0) and not bits[pid] & ~PARDES:
                grades.setdefault(f"pardes-only:{found}", 1)
        if cands:
            searches.append({"key": q["id"], "query": q["q"], "candidates": cands})
            judged[q["id"]] = {str(k): v for k, v in grades.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"searches": searches}, ensure_ascii=False) + "\n", "utf-8")
    args.judged.write_text(json.dumps(judged) + "\n", "utf-8")
    rel = sum(v for g in judged.values() for k, v in g.items() if not k.startswith("pardes-only:"))
    n = sum(1 for g in judged.values() for k in g if not k.startswith("pardes-only:"))
    print(f"{len(searches)} searches, {n} lines of ours judged, {rel} of them relevant")
    for k, v in why.most_common():
        print(f"  {k}: {v}")


def metrics(order: list, grades: dict) -> tuple[int, float, float]:
    first = grades.get(order[0], 0) if order else 0
    rr = next((1 / (i + 1) for i, d in enumerate(order[:10]) if grades.get(d)), 0.0)
    dcg = sum(grades.get(d, 0) / math.log2(i + 2) for i, d in enumerate(order[:10]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, sum(grades.values()))))
    return first, rr, dcg / ideal if ideal else 0.0


def score(args) -> None:
    judged = json.loads(args.judged.read_text("utf-8"))
    ranks = {}
    for item in args.ranks:
        name, _, path = item.partition("=")
        ranks[name] = json.loads(Path(path).read_text("utf-8"))
    names = list(ranks)
    per = defaultdict(dict)
    used = 0
    for key, g in judged.items():
        only_pardes = {int(k.split(":")[1]) for k in g if k.startswith("pardes-only:")}
        grades = {int(k): v for k, v in g.items() if not k.startswith("pardes-only:")}
        if args.without_pardes:
            grades = {d: v for d, v in grades.items() if d not in only_pardes}
        common = set(grades)
        for name in names:
            common &= set(ranks[name].get(key, {}).get("ids", []))
        if not any(grades[d] for d in common) or all(grades[d] for d in common):
            continue
        used += 1
        for name in names:
            order = [d for d in ranks[name][key]["ids"] if d in common]
            per[name][key] = metrics(order, grades)
    print(f"{used} searches with a relevant and a not relevant line that both rank"
          + (" (lines only Pardes's engines found left out)" if args.without_pardes else ""))
    for name in names:
        v = list(per[name].values())
        print(f"  {name}: first relevant {sum(x[0] for x in v)} ({100 * sum(x[0] for x in v) / len(v):.1f}%), "
              f"MRR@10 {sum(x[1] for x in v) / len(v):.3f}, nDCG@10 {sum(x[2] for x in v) / len(v):.3f}")
    if len(names) == 2:
        a, b = names
        keys = sorted(per[a])
        only_a = sum(per[a][k][0] and not per[b][k][0] for k in keys)
        only_b = sum(per[b][k][0] and not per[a][k][0] for k in keys)
        noise = 2 * math.sqrt(only_a + only_b)
        verdict = "above the noise" if abs(only_a - only_b) > noise else "within the noise"
        print(f"  first relevant, only {a} / only {b}: {only_a} / {only_b} (noise {noise:.1f}): {verdict}")
        d = [per[b][k][1] - per[a][k][1] for k in keys]
        rng = random.Random(5)
        boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
        print(f"  MRR@10, {b} minus {a}: {sum(d) / len(d):+.3f} [95%: {boots[50]:+.3f}, {boots[1949]:+.3f}]; "
              f"better in {sum(x > 0 for x in d)}, worse in {sum(x < 0 for x in d)}, the same in {sum(x == 0 for x in d)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("spec")
    a.add_argument("--rounds", type=Path, required=True)
    a.add_argument("--labels", type=Path, required=True)
    a.add_argument("--db", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--judged", type=Path, required=True)
    a.set_defaults(act=spec)
    b = sub.add_parser("score")
    b.add_argument("--judged", type=Path, required=True)
    b.add_argument("--ranks", action="append", required=True, help="<name>=<pool_rank.py's ranks file>")
    b.add_argument("--without-pardes", action="store_true")
    b.set_defaults(act=score)
    args = ap.parse_args()
    args.act(args)


if __name__ == "__main__":
    main()

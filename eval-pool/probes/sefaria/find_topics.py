"""Three ways of finding Sefaria's topic for a search, on the topics Pardes for desktop ships (its branch's
resources/sefaria/topics.tsv.gz, scripts/sefaria_content.py's rows):
- the name: the search is a topic's name or one of its other names, as typed (what the search's topic card does);
- the words: a topic's name stands in the search as whole words, its first word perhaps after one letter of ובכלמשה;
  the longest such name. Also when that name is half the search's words or more;
- the meaning: the topic whose name the pack's model puts nearest the search, at a cosine of 0.85, 0.8 or 0.7.

For each: how many of the searches users sent find a topic, and in the judged searches, whether the lines the topic is
about (its A rows, our lines by their reference) are the lines the judges found good, against the other judged lines.

    python eval-pool/probes/sefaria/find_topics.py build/sefaria/pardes-topics.tsv.gz [--db build/library/seforim.db]

Reads the history users sent (build/feedback/history.tsv), the judged searches' key (build/judge-pick/key.tsv) and
judge/grades.tsv. With --run (Eval.kt's run of the judged searches, build/judge/run.jsonl): the page the judges saw, its
first PAGE lines in the app's order, with the lines the search's topic is about put first. Whether the first line is
good, before and after, paired. Prints numbers only.
"""
import argparse
import collections
import csv
import gzip
import re
import sqlite3
import sys

import torch

sys.path[:0] = ["."]
import pool_rank  # noqa: E402
from pardes_semantic import model as m  # noqa: E402

MARKS = re.compile("[֑-ׇ]")
PREFIX = "ובכלמשה"
PAGE = 6        # the lines of a page the judges saw


def norm(s: str) -> str:
    return " ".join(re.sub(r"[\"'׳״.,:;!?()\[\]\-–—<>/]", " ", MARKS.sub("", s or "").replace("־", " ")).split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("topics")
    ap.add_argument("--db", default="build/library/seforim.db")
    ap.add_argument("--model", default="build/kaggle-output/tune3/tuned")
    ap.add_argument("--run", help="Eval.kt's run of the judged searches (build/judge/run.jsonl)")
    args = ap.parse_args()
    torch.set_num_threads(4)

    names, lines_of, refs = {}, collections.defaultdict(set), collections.defaultdict(set)
    for row in gzip.open(args.topics, "rt", encoding="utf-8"):
        f = row.rstrip("\n").split("\t")
        if f[0] == "T":
            names[int(f[1])] = [n for n in [norm(f[2])] + [norm(x) for x in f[3].split("|")] if len(n) >= 2]
        elif f[0] == "A":
            refs[f[2]].add(int(f[1]))
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    wanted = list(refs)
    for i in range(0, len(wanted), 900):
        chunk = wanted[i:i + 900]
        for lid, ref in con.execute(f"SELECT id, heRef FROM line WHERE heRef IN ({','.join('?' * len(chunk))})", chunk):
            for t in refs[ref]:
                lines_of[t].add(lid)
    topics = [t for t in names if lines_of[t] and names[t]]
    print(f"{len(topics)} topics with a Hebrew name and lines of ours; {sum(len(lines_of[t]) for t in topics):,} lines in all")
    size = {t: len(lines_of[t]) for t in topics}
    by_name = {}
    for t in sorted(topics, key=lambda t: -size[t]):
        for n in names[t]:
            by_name.setdefault(n, t)

    def by_words(query: str):
        words = query.split()
        best = None
        for i in range(len(words)):
            for j in range(len(words), i, -1):
                if best and (j - i) < best[0]:
                    break
                first = words[i]
                cands = [" ".join([first] + words[i + 1:j])]
                if len(first) > 2 and first[0] in PREFIX:
                    cands.append(" ".join([first[1:]] + words[i + 1:j]))
                for c in cands:
                    t = by_name.get(c)
                    if t is not None and (best is None or (j - i, size[t]) > (best[0], size[best[1]])):
                        best = (j - i, t)
        return best

    history = list(csv.DictReader(open("build/feedback/history.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    searches = sorted({norm(r["query"]) for r in history if norm(r.get("query"))})
    key = {r["qid"]: norm(r["query"]) for r in csv.DictReader(open("build/judge-pick/key.tsv", encoding="utf-8"),
                                                              delimiter="\t", quoting=csv.QUOTE_NONE)}
    grades = collections.defaultdict(dict)
    for r in csv.DictReader(open("judge/grades.tsv", encoding="utf-8"), delimiter="\t"):
        grades[r["qid"]][int(r["lineId"])] = int(r["grade"])
    judged = [q for q in grades if q in key and key[q]]
    pages = {}
    if args.run:
        import json
        for raw in open(args.run, encoding="utf-8"):
            r = json.loads(raw)
            pages[norm(r["query"])] = [x["lineId"] for x in r["hybrid"]][:PAGE]

    loaded = m.load("cpu", torch.float32, args.model)
    tv = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + names[t][0] for t in topics])
    texts = sorted(set(searches) | {key[q] for q in judged})
    qv = dict(zip(texts, pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + s for s in texts])))

    def nearest(query: str):
        sim = tv @ qv[query]
        k = int(sim.argmax())
        return float(sim[k]), topics[k]

    near = {s: nearest(s) for s in texts}
    methods = {
        "the name": lambda s: by_name.get(s),
        "the words": lambda s: (by_words(s) or (0, None))[1],
        "the words, half the search or more": lambda s: (lambda b: b[1] if b and 2 * b[0] >= len(s.split()) else None)(by_words(s)),
        "the meaning, cosine 0.85": lambda s: near[s][1] if near[s][0] >= 0.85 else None,
        "the meaning, cosine 0.8": lambda s: near[s][1] if near[s][0] >= 0.8 else None,
        "the meaning, cosine 0.7": lambda s: near[s][1] if near[s][0] >= 0.7 else None,
    }
    print(f"{len(searches)} distinct searches users sent; {len(judged)} judged searches")
    for name, find in methods.items():
        found = [s for s in searches if find(s) is not None]
        inside, outside, with_good, hits = [], [], 0, 0
        n = 0
        for q in judged:
            t = find(key[q])
            if t is None:
                continue
            n += 1
            mine = lines_of[t]
            inside += [g for line, g in grades[q].items() if line in mine]
            outside += [g for line, g in grades[q].items() if line not in mine]
            hits += any(line in mine for line in grades[q])
            with_good += any(g >= 2 and line in mine for line, g in grades[q].items())
        good_in = sum(g >= 2 for g in inside)
        good_out = sum(g >= 2 for g in outside)
        print(f"{name}: {len(found)} of {len(searches)} searches ({100 * len(found) / len(searches):.1f}%) | judged: {n} searches, "
              f"the topic's lines judged {len(inside)}, good {good_in} ({100 * good_in / max(1, len(inside)):.0f}%) | the other judged lines "
              f"{len(outside)}, good {good_out} ({100 * good_out / max(1, len(outside)):.0f}%) | a good line among the topic's in {with_good}")
        lengths = collections.Counter(min(len(s.split()), 4) for s in found)
        print("   by the search's words (1, 2, 3, 4+): " + ", ".join(str(lengths[k]) for k in (1, 2, 3, 4)))
        if pages:
            moved = only_before = only_after = 0
            for q in judged:
                t = find(key[q])
                page = pages.get(key[q])
                if t is None or not page:
                    continue
                mine = lines_of[t]
                after = [x for x in page if x in mine] + [x for x in page if x not in mine]
                if after[0] == page[0]:
                    continue
                moved += 1
                before_good = grades[q].get(page[0], 0) >= 2
                after_good = grades[q].get(after[0], 0) >= 2
                only_before += before_good and not after_good
                only_after += after_good and not before_good
            noise = 2 * (only_before + only_after) ** 0.5
            print(f"   the page with the topic's lines first: {moved} searches change their first line; good first only as it was "
                  f"{only_before}, only with the topic first {only_after} (noise {noise:.1f})")


if __name__ == "__main__":
    main()

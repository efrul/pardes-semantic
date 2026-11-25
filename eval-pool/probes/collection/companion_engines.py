"""The companion package's collected file (questions with passages marked as belonging or not): how many, and which
engine proposed each passage. A passage only Zayit's engine proposed comes from Zayit's semantic search, which this
repository uses nothing of (CLAUDE.md): such passages are counted apart, to be left out.

    python -I eval-pool/probes/collection/companion_engines.py build/companion/data/collected/train.jsonl
"""
import collections
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
engines, zayit_only, n = collections.Counter(), collections.Counter(), collections.Counter()
for r in rows:
    for kind in ("positives", "hard_negatives", "disputed", "easy_negatives"):
        for c in r[kind]:
            n[kind] += 1
            origin = [f.split("_")[0] for f in c.get("from", [])]
            for f in set(origin):
                engines[(kind, f)] += 1
            if origin and all(f == "zayit" for f in origin):
                zayit_only[kind] += 1
print(len(rows), "questions, by file:", dict(collections.Counter(r["source"] for r in rows)))
print("passages:", dict(n))
print("by engine:", sorted(engines.items(), key=lambda x: -x[1]))
print("proposed by Zayit's engine alone:", dict(zayit_only))

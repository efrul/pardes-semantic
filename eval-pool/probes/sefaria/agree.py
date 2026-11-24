"""Where a judged search is near a topic: are the topic's sources the lines the judges found good?"""
import collections, csv, pickle, re, sys
import bson, numpy as np, torch
sys.path[:0] = ["."]
import pool_rank
from pardes_semantic import model as m
torch.set_num_threads(4)
D = "build/sefaria/dump/dump/sefaria/"
MARKS = re.compile("[֑-ׇ]")
def norm(s):
    return " ".join(re.sub(r"[\"'׳״.,:;!?()\[\]\-–—<>/]", " ", MARKS.sub("", s).replace("־", " ")).split())
pairs = pickle.load(open("build/sefaria/about_pairs.pkl", "rb"))
he, sub = {}, {}
with open(D + "topics.bson", "rb") as f:
    for t in bson.decode_file_iter(f):
        pr = [x["text"] for x in t.get("titles", []) if x.get("lang") == "he" and x.get("primary")] or [x["text"] for x in t.get("titles", []) if x.get("lang") == "he"]
        if pr: he[t["slug"]] = norm(pr[0])
of_topic = collections.defaultdict(set)
for p in pairs:
    if p[1] != "ashlag-glossary": of_topic[p[0]].update(p[3])
slugs = sorted(s for s in of_topic if s in he)
loaded = m.load("cpu", torch.float32, "build/kaggle-output/tune3/tuned")
T = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + he[s] for s in slugs])
key = {r["qid"]: r["query"] for r in csv.DictReader(open("build/judge-pick/key.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)}
grades = {}
for r in csv.DictReader(open("judge/grades.tsv", encoding="utf-8"), delimiter="\t"):
    grades.setdefault(r["qid"], {})[int(r["lineId"])] = int(r["grade"])
qids = [q for q in grades if q in key]
S = pool_rank.embed_queries(loaded, [m.QUERY_PREFIX + norm(key[q]) for q in qids])
sim = S @ T.T
for th in (0.9, 0.85, 0.8, 0.7):
    near = [(q, slugs[int(sim[k].argmax())]) for k, q in enumerate(qids) if sim[k].max() >= th]
    inside = [(grades[q][l]) for q, s in near for l in grades[q] if l in of_topic[s]]
    outside = [(grades[q][l]) for q, s in near for l in grades[q] if l not in of_topic[s]]
    hit = sum(1 for q, s in near if any(l in of_topic[s] for l in grades[q]))
    good_known = sum(1 for q, s in near if any(g >= 2 and l in of_topic[s] for l, g in grades[q].items()))
    print(f"cosine {th}: {len(near)} of {len(qids)} judged searches near a topic | a judged line is among the topic's sources in {hit} | "
          f"judged lines among them: {len(inside)}, good {sum(g >= 2 for g in inside)} | the others: {len(outside)}, good {sum(g >= 2 for g in outside)} | "
          f"a good judged line is among them in {good_known} searches")

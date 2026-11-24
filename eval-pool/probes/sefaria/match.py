"""How many of the users' real searches are a topic of Sefaria's, by the words alone."""
import collections, csv, pickle, re, sys
import bson
D = "build/sefaria/dump/dump/sefaria/"
MARKS = re.compile("[֑-ׇ]")
def norm(s):
    s = MARKS.sub("", s).replace("־", " ")
    s = re.sub(r"[\"'׳״.,:;!?()\[\]\-]", "", s)
    return " ".join(s.split())
pairs = pickle.load(open("build/sefaria/about_pairs.pkl", "rb"))
per = collections.Counter(p[0] for p in pairs)
titles = collections.defaultdict(set)   # normalized Hebrew title -> topics
with open(D + "topics.bson", "rb") as f:
    for t in bson.decode_file_iter(f):
        if t["slug"] not in per: continue
        for x in t.get("titles", []):
            if x.get("lang") == "he" and x.get("text"):
                titles[norm(x["text"])].add(t["slug"])
print(len(per), "topics with a mapped source;", len(titles), "Hebrew titles")
def load(path, col="query", weight=None):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    return [(r[col], int(r[weight]) if weight and (r.get(weight) or "").isdigit() else 1) for r in rows if r.get(col)]
def report(name, qs):
    qs = [(norm(q), w) for q, w in qs if norm(q)]
    n = len(qs)
    exact = [q for q, _ in qs if q in titles]
    lens = collections.Counter(min(len(q.split()), 6) for q, _ in qs)
    by_len = collections.Counter(min(len(q.split()), 6) for q in exact)
    strip = [q for q, _ in qs if q not in titles and (q[1:] in titles and q[0] in "הבלמוכש")]
    print(f"{name}: {n} searches | exactly a topic: {len(exact)} ({100*len(exact)/n:.0f}%) | with a prefix letter off: +{len(strip)}")
    print("   by words (searches / a topic):", [(k, lens[k], by_len[k]) for k in sorted(lens)])
    srcs = sorted(per[next(iter(titles[q]))] for q in exact)
    if srcs: print("   sources of the matched topics: median", srcs[len(srcs)//2], "| with 5+:", sum(s >= 5 for s in srcs))
    return set(exact)
judged = load("build/judge-pick/key.tsv")
hist = load("build/feedback/history.tsv")
uniq = list({norm(q): (q, 1) for q, _ in hist}.values())
a = report("judged searches", judged)
b = report("history, each search once", uniq)

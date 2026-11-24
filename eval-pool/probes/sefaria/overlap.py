"""Do a topic's sources hold the topic's own words? By the origin of the link."""
import collections, pickle, random, re, sqlite3, sys
import bson
sys.path.insert(0, ".")
from pardes_semantic.clean import clean
D = "build/sefaria/dump/dump/sefaria/"
MARKS = re.compile("[֑-ׇ]")
def norm(s):
    s = MARKS.sub("", s).replace("־", " ")
    s = re.sub(r"[\"'׳״.,:;!?()\[\]\-–—<>/]", " ", s)
    return " ".join(s.split())
pairs = pickle.load(open("build/sefaria/about_pairs.pkl", "rb"))
he = {}
sub = {}
with open(D + "topics.bson", "rb") as f:
    for t in bson.decode_file_iter(f):
        ts = [x["text"] for x in t.get("titles", []) if x.get("lang") == "he" and x.get("text")]
        pr = [x["text"] for x in t.get("titles", []) if x.get("lang") == "he" and x.get("primary")]
        if ts:
            he[t["slug"]] = (norm((pr or ts)[0]), [norm(x) for x in ts])
        sub[t["slug"]] = t.get("subclass")
random.seed(3)
sample = random.sample(pairs, 30000)
con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
ids = sorted({p[3][0] for p in sample})
text = {}
for i in range(0, len(ids), 900):
    chunk = ids[i:i + 900]
    for line_id, content in con.execute(f"SELECT id, content FROM line WHERE id IN ({','.join('?' * len(chunk))})", chunk):
        text[line_id] = norm(clean(content or ""))
STOP = {"של", "על", "את", "דיני", "הלכות", "עם", "אל", "או", "בין", "פרשת"}
def holds(titles, line):
    toks = line.split()
    bare = set(toks) | {t[1:] for t in toks if len(t) > 3 and t[0] in "הבלמוכשד"} | {t[2:] for t in toks if len(t) > 4 and t[:2] in ("וה", "שה", "וב", "ול", "ומ", "של", "כש", "מה", "בה", "לה")}
    best = 0.0
    for title in titles:
        words = [w for w in title.split() if w not in STOP and len(w) > 1]
        if not words: continue
        best = max(best, sum(w in bare for w in words) / len(words))
    return best
by = collections.defaultdict(list)
lens = collections.defaultdict(list)
for topic, src, ref, lines, cp, tfidf, pr in sample:
    if topic not in he or lines[0] not in text: continue
    kind = "person" if sub.get(topic) in ("person", "author") else src
    by[kind].append(holds(he[topic][1], text[lines[0]]))
    lens[kind].append(len(text[lines[0]].split()))
print(f"{'origin':28s} pairs | none of the topic's words | some | all | words in the line (median)")
for kind, v in sorted(by.items(), key=lambda x: -len(x[1])):
    n = len(v); L = sorted(lens[kind])
    print(f"{kind:28s} {n:5d} | {100*sum(x == 0 for x in v)/n:4.0f}% | {100*sum(0 < x < 1 for x in v)/n:4.0f}% | {100*sum(x >= 1 for x in v)/n:4.0f}% | {L[len(L)//2]}")
allv = [x for v in by.values() for x in v]
print("all:", len(allv), f"none {100*sum(x == 0 for x in allv)/len(allv):.0f}%")

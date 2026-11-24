"""The 'about' links: by origin, the topics they hang on, the books they point to."""
import collections, pickle, re
import bson
D = "build/sefaria/dump/dump/sefaria/"
rows = pickle.load(open("build/sefaria/ref_topic_rows.pkl", "rb"))
cross = collections.Counter((r[2], r[3]) for r in rows)
print("origin x kind:", sorted(cross.items(), key=lambda x: -x[1]))
topics = {}
with open(D + "topics.bson", "rb") as f:
    for t in bson.decode_file_iter(f):
        topics[t["slug"]] = t
def he_title(slug):
    t = topics.get(slug) or {}
    for x in t.get("titles", []):
        if x.get("lang") == "he" and x.get("primary"):
            return x["text"]
    for x in t.get("titles", []):
        if x.get("lang") == "he":
            return x["text"]
    return None
about = [r for r in rows if r[3] == "about"]
slugs = {r[0] for r in about}
with_he = {s for s in slugs if he_title(s)}
print("about links", len(about), "| topics", len(slugs), "| with a Hebrew title", len(with_he), "| links on those", sum(r[0] in with_he for r in about))
sub = collections.Counter((topics.get(s) or {}).get("subclass") for s in slugs)
print("topic subclasses:", sub.most_common())
per = collections.Counter(r[0] for r in about)
v = sorted(per.values())
print("about sources a topic: median", v[len(v)//2], "| 5+", sum(x >= 5 for x in v), "| 20+", sum(x >= 20 for x in v), "| 100+", sum(x >= 100 for x in v))
def book(ref):
    m = re.match(r"^(.*?)(?:,? \d+[ab]?(?::\d+)*(?:-\d+[ab]?(?::\d+)*)?)?$", ref)
    return (m.group(1) if m else ref)
books = collections.Counter(book(r[1]) for r in about)
print("books referenced:", len(books), "| top:", books.most_common(40))
words = collections.Counter(len(he_title(s).split()) for s in with_he)
print("words in the Hebrew title:", sorted(words.items()))
big = sorted(with_he, key=lambda s: -per[s])[:40]
print("largest:", [(he_title(s), per[s]) for s in big])
import random
random.seed(5)
print("random:", [(he_title(s), per[s]) for s in random.sample(sorted(with_he), 60)])

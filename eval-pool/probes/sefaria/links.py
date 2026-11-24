"""Statistics of the topic links in Sefaria's dump: the links from a topic to a source."""
import collections, json, pickle, sys
import bson
D = "build/sefaria/dump/dump/sefaria/"
def docs(name):
    with open(D + name + ".bson", "rb") as f:
        yield from bson.decode_file_iter(f)
cls = collections.Counter(); keys = collections.Counter()
by_source = collections.Counter(); by_type = collections.Counter(); sheet = collections.Counter()
desc = collections.Counter(); generated = collections.Counter(); expanded = []; per_topic = collections.Counter()
per_topic_src = collections.defaultdict(collections.Counter)
example = {}
rows = []
n = 0
for d in docs("topic_links"):
    n += 1
    c = d.get("class"); cls[c] += 1
    if c != "refTopic":
        continue
    for k in d: keys[k] += 1
    is_sheet = bool(d.get("is_sheet"))
    sheet[is_sheet] += 1
    if is_sheet:
        continue
    src = d.get("dataSource"); by_source[src] += 1
    by_type[d.get("linkType")] += 1
    generated[d.get("generatedBy")] += 1
    ds = d.get("descriptions") or {}
    he = (ds.get("he") or {}); en = (ds.get("en") or {})
    desc[(bool(he.get("title") or he.get("prompt")), bool(en.get("title") or en.get("prompt")))] += 1
    ex = d.get("expandedRefs") or []
    expanded.append(len(ex))
    per_topic[d.get("toTopic")] += 1
    per_topic_src[src][d.get("toTopic")] += 1
    example.setdefault(src, d)
    order = d.get("order") or {}
    rows.append((d.get("toTopic"), d.get("ref"), src, d.get("linkType"), len(ex), (he.get("title") or ""), (he.get("prompt") or ""),
                 order.get("curatedPrimacy"), order.get("tfidf"), order.get("pr"), order.get("numDatasource"), d.get("generatedBy")))
print("documents", n, "| classes", dict(cls))
print("refTopic keys", keys.most_common(30))
print("sheets vs texts", dict(sheet))
print("by data source", by_source.most_common())
print("by link type", by_type.most_common())
print("generatedBy", generated.most_common(8))
print("descriptions (he, en)", dict(desc))
expanded.sort()
print("segments a link: median", expanded[len(expanded)//2], "mean", round(sum(expanded)/len(expanded), 1), "p90", expanded[int(len(expanded)*.9)], "max", expanded[-1], "| links with 0", sum(1 for x in expanded if x == 0))
v = sorted(per_topic.values())
print("topics with a text source:", len(v), "| sources a topic: median", v[len(v)//2], "mean", round(sum(v)/len(v), 1), "| with 5+", sum(x >= 5 for x in v), "20+", sum(x >= 20 for x in v), "100+", sum(x >= 100 for x in v))
for src, c in per_topic_src.items():
    print("  ", src, "topics", len(c), "links", sum(c.values()))
for src, d in example.items():
    print("example", src, json.dumps({k: v for k, v in d.items() if k not in ("_id", "expandedRefs")}, ensure_ascii=False, default=str)[:700])
pickle.dump(rows, open("build/sefaria/ref_topic_rows.pkl", "wb"))

"""Sefaria's topic links mapped to our library's lines, by the reference the library's generator gave each line."""
import collections, json, pickle, re, sqlite3, sys, time
import bson
D = "build/sefaria/dump/dump/sefaria/"

def gematria(n):
    if n <= 0: return str(n)
    out = ""
    th, r = divmod(n, 1000)
    if th: out += gematria(th) + " "
    for v, l in ((400, "ת"), (300, "ש"), (200, "ר"), (100, "ק")):
        while r >= v: out += l; r -= v
    if r == 15: out += "טו"; r = 0
    elif r == 16: out += "טז"; r = 0
    for v, l in ((90, "צ"), (80, "פ"), (70, "ע"), (60, "ס"), (50, "נ"), (40, "מ"), (30, "ל"), (20, "כ"), (10, "י")):
        if r >= v: out += l; r -= v
    for v, l in ((9, "ט"), (8, "ח"), (7, "ז"), (6, "ו"), (5, "ה"), (4, "ד"), (3, "ג"), (2, "ב"), (1, "א")):
        if r >= v: out += l; r -= v
    return out

def primary(node, lang):
    for t in node.get("titles", []):
        if t.get("lang") == lang and t.get("primary"):
            return t["text"]
    return ""

leaves = {}   # the English prefix of a leaf -> (our Hebrew prefix, address types)
with open(D + "index.bson", "rb") as f:
    for d in bson.decode_file_iter(f):
        s = d.get("schema") or {}
        en, he = d.get("title") or primary(s, "en"), primary(s, "he")
        if not en or not he: continue
        if "nodes" not in s:
            leaves[en] = (he + " ", s.get("addressTypes") or [])
            continue
        def walk(node, en_prefix, he_prefix):
            for c in node.get("nodes", []):
                default = c.get("default") or c.get("key") == "default"
                ce, ch = primary(c, "en"), primary(c, "he")
                e2 = en_prefix if default or not ce else en_prefix + ", " + ce
                h2 = he_prefix if default or not ch else he_prefix + ch + ", "
                if "nodes" in c:
                    walk(c, e2, h2)
                else:
                    leaves.setdefault(e2, (h2 + " ", c.get("addressTypes") or []))
        walk(s, en, he + ", ")
print(len(leaves), "leaves of the index", flush=True)

SECTIONS = re.compile(r"^(.*?) (\d+[ab]?(?::\d+[ab]?)*)$")
def our_ref(ref):
    m = SECTIONS.match(ref)
    if not m or m.group(1) not in leaves:
        return None
    he_prefix, types = leaves[m.group(1)]
    parts = []
    for i, sec in enumerate(m.group(2).split(":")):
        kind = types[i] if i < len(types) else "Integer"
        if sec[-1] in "ab":
            if kind not in ("Talmud", "Folio"): return None
            parts.append(gematria(int(sec[:-1])) + ("." if sec[-1] == "a" else ":"))
        elif kind == "Talmud":
            return None
        else:
            parts.append(gematria(int(sec)))
    return he_prefix + ", ".join(parts)

t0 = time.time()
con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
ours = {}
for line_id, he in con.execute("SELECT id, heRef FROM line WHERE heRef IS NOT NULL"):
    ours.setdefault(he, line_id)
print(len(ours), "references of our lines,", round(time.time() - t0), "s", flush=True)

stats = collections.defaultdict(collections.Counter)
missing_prefix = collections.Counter(); missing_line = collections.Counter()
pairs = []   # (topic, data source, link's ref, [our line ids], curatedPrimacy he, tfidf)
with open(D + "topic_links.bson", "rb") as f:
    for d in bson.decode_file_iter(f):
        if d.get("class") != "refTopic" or d.get("is_sheet") or d.get("linkType") != "about":
            continue
        src = d.get("dataSource")
        refs = d.get("expandedRefs") or []
        found = []
        for r in refs:
            h = our_ref(r)
            if h is None:
                m = SECTIONS.match(r)
                missing_prefix[m.group(1) if m else r] += 1
                stats[src]["segment: no such part in the index"] += 1
            elif h in ours:
                found.append(ours[h]); stats[src]["segment: found"] += 1
            else:
                missing_line[SECTIONS.match(r).group(1)] += 1
                stats[src]["segment: not in our library"] += 1
        stats[src]["links"] += 1
        stats[src]["links with a line"] += bool(found)
        stats[src]["links with all their lines"] += bool(refs) and len(found) == len(refs)
        if found:
            order = d.get("order") or {}
            cp = (order.get("curatedPrimacy") or {}).get("he")
            pairs.append((d["toTopic"], src, d.get("ref"), found, cp, order.get("tfidf"), order.get("pr")))
for src, c in stats.items():
    print(src, dict(c))
tot = collections.Counter()
for c in stats.values(): tot.update(c)
print("all", dict(tot))
print("parts not in the index:", missing_prefix.most_common(15))
print("books not in our library:", missing_line.most_common(25))
print("pairs with a line:", len(pairs), "| topics:", len({p[0] for p in pairs}), "| distinct lines:", len({l for p in pairs for l in p[3]}))
pickle.dump(pairs, open("build/sefaria/about_pairs.pkl", "wb"))

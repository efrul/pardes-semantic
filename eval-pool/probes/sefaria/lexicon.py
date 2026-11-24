"""The dictionaries of Sefaria's dump against the Bavli's own words: how much of the Talmud a dictionary's headwords
cover, a word as it stands or without a letter or two before it.

    python -I eval-pool/probes/sefaria/lexicon.py
"""
import collections, re, sqlite3
import bson
D = "build/sefaria/dump/dump/sefaria/"
MARKS = re.compile("[֑-ׇ]")
PRE = "דובלכמקהש"


def norm(s):
    return re.sub(r"[^א-ת ]", "", MARKS.sub("", s or "")).strip()


def flat(v):
    if isinstance(v, str):
        return re.sub(r"<[^>]+>", " ", v)
    if isinstance(v, dict):
        return " ".join(flat(x) for k, x in v.items() if k not in ("grammar", "number", "refs"))
    if isinstance(v, list):
        return " ".join(flat(x) for x in v)
    return ""


heads = collections.defaultdict(set)
words = collections.defaultdict(list)
with open(D + "lexicon_entry.bson", "rb") as f:
    for d in bson.decode_file_iter(f):
        name = d.get("parent_lexicon")
        if norm(d.get("headword")):
            heads[name].add(norm(d.get("headword")))
        words[name].append(len(flat(d.get("content")).split()))
con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
bavli = con.execute("SELECT id FROM category WHERE title = 'בבלי'").fetchone()[0]
books = [r[0] for r in con.execute(
    "SELECT b.id FROM book b WHERE b.sourceId = 1 AND b.isBaseBook = 1 AND b.categoryId IN "
    "(SELECT descendantId FROM category_closure WHERE ancestorId = ?)", (bavli,))]
freq = collections.Counter()
for b in books:
    for (content,) in con.execute("SELECT content FROM line WHERE bookId = ? AND heRef IS NOT NULL", (b,)):
        freq.update(norm(re.sub(r"<[^>]+>", " ", content or "")).split())
total = sum(freq.values())
print(f"{len(books)} tractates: {total} running words, {len(freq)} distinct")


def covered(w, hs):
    return w in hs or (len(w) > 2 and w[0] in PRE and w[1:] in hs) or (len(w) > 3 and w[0] in PRE and w[1] in PRE and w[2:] in hs)


for name, hs in sorted(heads.items(), key=lambda x: -len(x[1])):
    tok = sum(c for w, c in freq.items() if covered(w, hs))
    typ = sum(1 for w in freq if covered(w, hs))
    v = sorted(words[name])
    print(f"{name}: {len(words[name])} entries, {len(hs)} headwords, {v[len(v) // 2]} words an entry | of the Bavli: "
          f"{100 * tok / total:.0f}% of the running words, {100 * typ / len(freq):.0f}% of the distinct")

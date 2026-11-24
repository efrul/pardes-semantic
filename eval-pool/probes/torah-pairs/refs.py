"""Torah-pairs' passages mapped to our lines by their Hebrew reference, whatever the book's id: how many, and is the text ours?"""
import collections, json, pickle, random, re, sqlite3, sys
sys.path.insert(0, ".")
from pardes_semantic.clean import clean
MARKS = re.compile("[֑-ׇ]")
def norm(s):
    return " ".join(re.sub(r"[^א-ת ]", " ", MARKS.sub("", clean(s or ""))).split())
def key(ref):
    return re.sub(r"[ ,]", "", ref or "")
con = sqlite3.connect("file:build/library/seforim.db?mode=ro", uri=True)
ours = {}
for line_id, he in con.execute("SELECT id, heRef FROM line WHERE heRef IS NOT NULL"):
    ours.setdefault(key(he), line_id)
print(len(ours), "references of ours", flush=True)
random.seed(4)
out = {}
for name, qf in (("torahqa.clean", "question"), ("micropedia", "query"), ("et", "query"), ("shut_passages", "query"), ("mafteach", "abstract")):
    n = found = same = checked = 0
    conf = collections.Counter(); roles = collections.Counter(); books = collections.Counter(); qlen = []
    rows = []
    with open(f"build/torah-pairs/{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); n += 1
            p = d.get("passage") if isinstance(d.get("passage"), dict) else {}
            line_id = ours.get(key(p.get("he_ref")))
            if line_id is None: continue
            found += 1
            conf[p.get("confidence")] += 1; roles[p.get("role") or p.get("match_kind")] += 1; books[p.get("book")] += 1
            q = d.get(qf) or ""
            qlen.append(len(q.split()))
            rows.append((d.get("id") or f"{name}:{n}", q, line_id, p.get("confidence"), p.get("margin"), d.get("answer"), p.get("line_end"), p.get("line_index")))
            if random.random() < 0.01:
                checked += 1
                ours_text = norm(con.execute("SELECT content FROM line WHERE id = ?", (line_id,)).fetchone()[0])
                a = norm(p.get("text"))[:40]
                same += bool(a) and (a in ours_text or ours_text[:40] in norm(p.get("text")))
    qlen.sort()
    print(f"== {name}: {n} rows | on our lines {found} ({100*found/n:.0f}%) | text agrees in {same} of {checked} checked | distinct lines {len({r[2] for r in rows})} | books {len(books)}")
    print("   confidence:", conf.most_common(5), "| roles:", roles.most_common(6))
    print("   words in the query: median", qlen[len(qlen)//2] if qlen else None, "| top books:", [b for b, _ in books.most_common(8)])
    out[name] = rows
pickle.dump(out, open("build/torah-pairs/mapped.pkl", "wb"))

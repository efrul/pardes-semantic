"""Writes the cases that BpeTokenizerParityTest checks the app's reading of a query and a line, and its tokenizer of
the reranker, against.

    python reranker/make_fixtures.py --reranker build/pardes-reranker --db seforim.db \
        --out pardes/SeforimApp/src/jvmTest/resources/semantic   # then committed in Pardes

Each line of reranker-parity.jsonl.gz: {"query": raw, "line": raw, "queryText": query_text(query), "lineText":
line_text(line), "ids": the pair's ids as the app feeds the model}. The pair is `<bos> query <eos> line <eos>` of at
most the manifest's maxTokens, the line giving way first, as the tokenizers library truncates the second of a pair;
the app also cuts a query to queryMaxTokens before that, and so do the cases. The texts are the samples, edge cases,
and (with --db) lines drawn from the base books only, as make_parity_fixtures.py draws them: public-domain texts.
The reranker's tokenizer.json is copied gzipped next to them, as reranker-tokenizer.json.gz.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from build_reranker import SPEC_NAME, line_text, query_text  # noqa: E402
from make_parity_fixtures import corpus_lines  # noqa: E402
from pardes_semantic.samples import EXTRA, PAIRS  # noqa: E402

EDGE_LINES = [
    "", " ", "(א) בראשית ברא אלהים", "(קיט) אשרי תמימי דרך", "(הגה) ויש אומרים", "(12) שורה עם מספר", "(א') גרש במספר",
    "בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃", "כָּל־הַנְּשָׁמָה תְּהַלֵּל יָהּ", "מקף־עברי ופסק׀ וסוף פסוק׃ ונקודתיים:",
    "ראשי תיבות: רש\"י, רמב\"ם, שו\"ע, חז\"ל, וכשה\"ק, בגמ' ובתוס'", "<a href=\"x\">קישור</a><br><i>נטוי</i> &amp; &nbsp;עוד",
    "emoji 😀 and 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 and 中文 and ΣΊΣΥΦΟΣ İstanbul ß ﬁ Ⅻ ①", "x" * 150 + " " + "א" * 120,
    "1,000,000.50 ו-3.14 ו-ב' ו-ב", "The quick brown fox, tokenization and Wissenschaft des Judentums", "a" + "́" + "é",
    "▁ כבר ▁▁ בטקסט", "מילה " * 400,
]
EDGE_QUERIES = ["", "שבת", "מה דין מי ששכח", "האם מותר לטלטל מוקצה בשבת לצורך גופו", "כָּל־הַנְּשָׁמָה", "what is muktzeh", "שאלה " * 90]


def pair_ids(tok: Tokenizer, query: str, line: str, max_tokens: int, query_max: int) -> list[int]:
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    first = tok.encode(query, add_special_tokens=False).ids[:query_max]
    second = tok.encode(line, add_special_tokens=False).ids[:max_tokens - 3 - len(first)]
    return [bos, *first, eos, *second, eos]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranker", type=Path, required=True, help="reranker/build_reranker.py's folder")
    ap.add_argument("--db", type=Path)
    ap.add_argument("--lines", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=5786)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads((args.reranker / SPEC_NAME).read_text("utf-8"))
    max_tokens, query_max = manifest["maxTokens"], manifest["queryMaxTokens"]
    tokenizer_path = args.reranker / manifest["tokenizer"]
    tok = Tokenizer.from_file(str(tokenizer_path))
    library = Tokenizer.from_file(str(tokenizer_path))
    library.enable_truncation(max_length=max_tokens, strategy="only_second")

    lines = EDGE_LINES + [p for _, p in PAIRS] + EXTRA
    if args.db:
        lines += corpus_lines(args.db, args.lines, args.seed)
    # A query for each line: the samples' and the edge cases' in turn, and now and then words of a line
    queries = EDGE_QUERIES + [q for q, _ in PAIRS]
    cases, checked = [], 0
    for i, line in enumerate(lines):
        query = " ".join(line_text(lines[i - 1]).split()[2:7]) if i % 3 == 2 else queries[i % len(queries)]
        q, t = query_text(query), line_text(line)
        ids = pair_ids(tok, q, t, max_tokens, query_max)
        # Where the query is not cut, the tokenizers library's own pair
        if t and len(tok.encode(q, add_special_tokens=False).ids) <= query_max:
            assert library.encode(q, t).ids == ids, (q, t[:60])
            checked += 1
        cases.append({"query": query, "line": line, "queryText": q, "lineText": t, "ids": ids})

    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out / "reranker-parity.jsonl.gz", "wt", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    with open(tokenizer_path, "rb") as src, gzip.open(args.out / "reranker-tokenizer.json.gz", "wb") as dst:
        dst.write(src.read())
    longest = max(len(c["ids"]) for c in cases)
    print(f"{len(cases)} cases ({checked} checked against the tokenizers library's pair, the longest {longest} tokens) -> {args.out}")


if __name__ == "__main__":
    main()

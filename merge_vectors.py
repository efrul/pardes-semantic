"""Patches an index's vectors with those of kaggle_run.py push patch: the lines that merging short lines changed get the
patch's vectors, the short lines it left out lose theirs, and every other line keeps its own. A whole library embedded
again is five hours of GPU; the lines that change are a quarter of it.

    python merge_vectors.py --base build/kaggle-output/int8/vectors-int8 \
        --patch build/kaggle-output/patch/vectors-patch-int8 --out build/vectors-merged

Both in quantize_vectors.py's format (ids.i64, vecs.i8, meta.txt, embed-info.json) and of the same model and cleaner;
the patch also holds short-ids.i64 (embed_corpus.py --short-merge). Writes the same format, for VectorIndexBuilder
--vectors, in the base's order.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CHUNK = 500_000  # rows at a time: 500k x 384 int8 = 192 MB


def read(path: Path) -> tuple[np.ndarray, np.ndarray, int, dict]:
    count, dim = (int(x) for x in (path / "meta.txt").read_text().split())
    ids = np.fromfile(path / "ids.i64", dtype="<i8")
    vecs = np.memmap(path / "vecs.i8", dtype=np.int8, mode="r", shape=(count, dim))
    if len(ids) != count:
        sys.exit(f"{path}: {len(ids)} ids, meta.txt says {count}")
    return ids, vecs, dim, json.loads((path / "embed-info.json").read_text("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="the index's vectors (int8)")
    ap.add_argument("--patch", type=Path, required=True, help="kaggle_run.py push patch's vectors-patch-int8")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base_ids, base_vecs, dim, base_info = read(args.base)
    patch_ids, patch_vecs, patch_dim, patch_info = read(args.patch)
    for key in ("model", "cleaner"):
        if base_info.get(key) != patch_info.get(key):
            sys.exit(f"Not the same {key}: {base_info.get(key)} and {patch_info.get(key)}")
    if dim != patch_dim:
        sys.exit(f"Not the same dimension: {dim} and {patch_dim}")
    short = np.fromfile(args.patch / "short-ids.i64", dtype="<i8")

    keep = ~np.isin(base_ids, short)
    replace = np.isin(base_ids, patch_ids)
    order = np.argsort(patch_ids, kind="stable")
    sorted_patch = patch_ids[order]
    unknown = int((~np.isin(patch_ids, base_ids)).sum())
    print(f"base {len(base_ids)} vectors; patch {len(patch_ids)} ({int(replace.sum())} of the base's lines, {unknown} "
          f"not in it); short lines {len(short)} ({int((~keep).sum())} of the base's lines)", flush=True)
    if unknown:
        sys.exit("The patch holds lines the base does not: not the same library?")
    if (replace & ~keep).any():
        sys.exit("A line both patched and left out")

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    with (args.out / "ids.i64").open("wb") as fi, (args.out / "vecs.i8").open("wb") as fv:
        for i in range(0, len(base_ids), CHUNK):
            ids = base_ids[i:i + CHUNK]
            vecs = np.array(base_vecs[i:i + CHUNK])
            r = replace[i:i + CHUNK]
            if r.any():
                vecs[r] = patch_vecs[order[np.searchsorted(sorted_patch, ids[r])]]
            k = keep[i:i + CHUNK]
            fi.write(ids[k].astype("<i8").tobytes())
            fv.write(vecs[k].tobytes())
            written += int(k.sum())
    (args.out / "meta.txt").write_text(f"{written} {dim}\n")
    info = dict(base_info)
    info |= {"shortMerge": patch_info.get("shortMerge"), "mergeLead": patch_info.get("mergeLead"),
             "patchedRows": int(replace.sum()), "shortLinesLeftOut": int((~keep).sum()),
             "patchBackend": patch_info.get("backend"), "patchSource": patch_info.get("source")}
    (args.out / "embed-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"{written} vectors -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

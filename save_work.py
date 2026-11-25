"""Keeps the working files that no repository holds in one private Kaggle dataset, `<user>/pardes-semantic-work`: the
pairs a run learns, the tests kept out, what the miner found, a reranker's scores, the evaluations' ranks. A session's
container is reclaimed when it ends, and these files took hours to make (the skill pardes-semantic-search, Saving the
working data).

    python save_work.py build/train/micropedia-pairs.jsonl build/train/micropedia-pairs-test.jsonl -m "what they are"
    python save_work.py build/kaggle-output/pooln/ranks-*.json --prefix pooln- -m "..."   # names that could clash
    python save_work.py --list                     # what the dataset holds
    python save_work.py --restore build/work       # every file of it, back into a folder

A file is kept under its own name (with --prefix before it); a later file of the same name replaces it, and Kaggle
keeps the dataset's earlier versions. build/work-mirror holds every file of the dataset: each save uploads it whole as
the next version, so a mirror that a new container lacks is downloaded first, file by file (a private dataset has no
whole download). MANIFEST.tsv in it says where each file came from and when.

Never a user's data: no file of the feedback's inbox or of a folder of the feedback's (packages, histories, grades) is
uploaded (refused). Nothing here names the account: it comes from KAGGLE_USERNAME, as kaggle_run.py's.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kaggle_run import kaggle, user  # noqa: E402

NAME = "pardes-semantic-work"
MIRROR = HERE / "build" / "work-mirror"
MANIFEST = "MANIFEST.tsv"


def refused(path: Path) -> bool:
    """A user's data: the feedback's inbox, or a folder of the feedback's (build/feedback, build/feedback-1124)."""
    parts = [p.lower() for p in path.resolve().parts]
    return any("feedback-inbox" in p or p.startswith("feedback") for p in parts[:-1])


def dataset() -> str:
    return f"{user()}/{NAME}"


def exists() -> bool:
    r = subprocess.run(["kaggle", "datasets", "status", dataset()], capture_output=True, text=True)
    return r.returncode == 0 and "ready" in r.stdout


def remote_files() -> list[str]:
    """The dataset's files, every page of them."""
    names, token = [], None
    while True:
        args = ["datasets", "files", dataset(), "--csv", "--page-size", "200"] + (["--page-token", token] if token else [])
        out = kaggle(*args, capture=True)
        token = None
        for line in out.splitlines():
            if line.startswith("Next Page Token"):
                token = line.split("=", 1)[1].strip()
            elif line and not line.startswith("name,") and "," in line:
                names.append(line.split(",", 1)[0])
        if not token:
            return names


def download(name: str, into: Path) -> None:
    kaggle("datasets", "download", dataset(), "-f", name, "-p", str(into), "--force", capture=True)
    packed = into / f"{name}.zip"
    if packed.exists() and not (into / name).exists():   # a large file comes zipped
        with zipfile.ZipFile(packed) as z:
            z.extractall(into)
        packed.unlink()


def mirror_complete() -> None:
    """build/work-mirror with every file the dataset has (a new container has none of them)."""
    MIRROR.mkdir(parents=True, exist_ok=True)
    if not exists():
        return
    missing = [n for n in remote_files() if not (MIRROR / n).exists()]
    for k, name in enumerate(missing, 1):
        print(f"downloading {k}/{len(missing)}: {name}", flush=True)
        download(name, MIRROR)


def save(paths: list[Path], prefix: str, message: str) -> None:
    for p in paths:
        if not p.is_file():
            sys.exit(f"Not a file: {p}")
        if refused(p):
            sys.exit(f"Refused: {p} is a user's data (the feedback's), which stays out of every upload")
    mirror_complete()
    rows = (MIRROR / MANIFEST).read_text("utf-8").splitlines()[1:] if (MIRROR / MANIFEST).exists() else []
    kept = {r.split("\t", 1)[0]: r for r in rows}
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    for p in paths:
        name = prefix + p.name
        target = MIRROR / name
        if target.exists():
            target.unlink()
        try:
            os.link(p, target)
        except OSError:
            shutil.copy2(p, target)
        try:
            source = str(p.resolve().relative_to(HERE))
        except ValueError:
            source = p.name
        kept[name] = "\t".join([name, str(p.stat().st_size), source, stamp, message.replace("\t", " ")])
    (MIRROR / MANIFEST).write_text("name\tbytes\tfrom\tsaved\twhat\n" + "".join(r + "\n" for r in sorted(kept.values())),
                                   "utf-8")
    meta = MIRROR / "dataset-metadata.json"
    meta.write_text('{"title": "%s", "id": "%s", "licenses": [{"name": "other"}]}\n' % (NAME, dataset()), "utf-8")
    try:
        if exists():
            kaggle("datasets", "version", "-p", str(MIRROR), "-m", message or stamp, capture=True)
        else:
            kaggle("datasets", "create", "-p", str(MIRROR), capture=True)   # private, as every dataset of the account's
    finally:
        meta.unlink()
    for _ in range(90):
        time.sleep(10)
        if exists():
            break
    n = len(remote_files())
    print(f"{len(paths)} saved; the dataset holds {n} files ({NAME}, private)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--prefix", default="", help="before each file's name, for names that could clash (pooln-)")
    ap.add_argument("-m", "--message", default="", help="what the files are, for the manifest and the version")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--restore", type=Path, help="download every file of the dataset into this folder")
    args = ap.parse_args()
    if args.list:
        for name in remote_files():
            print(name)
    elif args.restore:
        args.restore.mkdir(parents=True, exist_ok=True)
        for name in remote_files():
            if not (args.restore / name).exists():
                print(f"downloading {name}", flush=True)
                download(name, args.restore)
    elif args.files:
        save(args.files, args.prefix, args.message)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

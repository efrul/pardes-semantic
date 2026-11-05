"""A trained model made a release of this repository (models.yml): the folder its Kaggle stage left (kaggle_run.py),
checked and zipped, with the release's tag, title and notes. The model is named by its id, the one the app's
manifest and the vector index carry (pardes_semantic.model.model_id): the folder's name, then the first 7 hex digits
of the sha256 of its weights, then the cleaner.

    python release_model.py --source-of student          # "distill student": the stage and its output folder
    python release_model.py --model student --folder dl/student --out release --meta meta [--version 12] [--note …]

release/: <tag>.zip, or <tag>.zip.001, .002… under GitHub's 2 GiB per file, and SHA256SUMS: the release's files.
meta/: tag.txt, title.txt and notes.md. The Kaggle account is never named in what is released: with its name in
$KAGGLE_OWNER, a path or a kernel's name that holds it is rewritten to <owner> in a text file, and anywhere else (a
binary file, a file's name, the note) it stops the release.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pardes_semantic.clean import CLEANER_ID  # noqa: E402

# The model: its Kaggle stage, the folder of the stage's output, the folder in the zip, and its name on the release
KINDS = {
    "teacher": ("train", "teacher", "teacher", "המורה"),
    "tuned": ("tune", "tuned", "tuned", "המורה אחרי השאלות"),
    "student": ("distill", "student", "student", "התלמיד"),
    "student-onnx": ("embed", "model-student", "pardes-model", "התלמיד ל־ONNX"),
}

# Under GitHub's 2 GiB (2,147,483,648 bytes) per release file
PART_BYTES = 2_000_000_000

TEXT = {".json", ".md", ".txt", ".py", ".yaml", ".yml", ".jinja", ".cfg", ".ini", ".csv", ".tsv"}
OWNER_TEXT = "<owner>"

DESCRIPTION = {
    "teacher": "המודל של דיקטה, neodictabert-bilingual-embed, מאומן על זוגות מהקישורים של הספרייה: שלב `train` של "
               "`kaggle_run.py`.",
    "tuned": "המורה של `train` אחרי אימון נוסף על שאלות האימון של train-v2 ועל חלק מזוגות הקישורים: שלב `tune`. ממנו "
             "מזקקים תלמיד (`distill --teacher tuned`).",
    "student": "המורה מזוקק למודל קטן ומהיר יותר: שלב `distill`.",
    "student-onnx": "התלמיד כמו שפרדס מריץ אותו, מיוצא ב־ONNX עם משקלים ב־int8, עם `tokenizer.json` ו־`pardes-model.json`: "
                    "שלב `embed`. זו התיקייה `pardes-model` של קובצי החיפוש לפי משמעות של פרדס שיש להם אותו מזהה.",
}

USE_TORCH = """\
**שימוש**
```python
# pip install -r requirements.txt, בריפו הזה
from pardes_semantic import model as m
from pardes_semantic.clean import clean

loaded = m.load(source="{inner}")      # התיקייה מתוך ה־zip
q = m.embed(loaded, [m.QUERY_PREFIX + clean("מי שאכל ושכח ולא בירך")], m.QUERY_MAX_TOKENS)
d = m.embed(loaded, [m.DOCUMENT_PREFIX + clean(t) for t in texts], m.DOCUMENT_MAX_TOKENS)
scores = q @ d.T                       # הווקטורים מנורמלים
```
השאילתה מתחילה ב־`query: ` והשורות בלי תחילית. שתיהן עוברות קודם את `pardes_semantic/clean.py`, כמו באימון ובפרדס. \
`export_onnx.py --source {inner}` מייצא אותו למודל שפרדס טוען.
"""

USE_ONNX = """\
**שימוש**
פרדס משתמש במודל רק עם אינדקס וקטורים שהוטמע בו, עם אותו מזהה. להתקנה בפרדס מורידים את קובצי החיפוש לפי משמעות \
מ[עמוד הגרסאות של פרדס](https://github.com/efrul/Pardes/releases), או מתוך היישום: הגדרות › ניהול נתונים › חיפוש לפי \
משמעות. בריפו הזה הוא נמדד ב־`embed_corpus.py --backend onnx --model pardes-model` וב־`evaluate.py`.
"""

LICENCE = """\
**רישיון**
המודל נגזר מ־[neodictabert-bilingual-embed](https://huggingface.co/dicta-il/neodictabert-bilingual-embed) של דיקטה: \
המרכז הישראלי לניתוח טקסטים, ומופץ לפי [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). מי שמעביר אותו הלאה \
מעביר גם את `{inner}/NOTICE.md`. דיקטה אינה ממליצה על פרדס. המודל אומן על הטקסטים של הספרייה (SeforimLibrary, מספריא \
ומאוצריא), והרישיונות של הטקסטים הם של המקורות שלהם. טקסט הבבלי שבספרייה, למשל, הוא ב־CC BY-NC.

פרדס מבוסס על [זית](https://github.com/kdroidFilter/Zayit) מאת אליהו גמבש ואינו מטעמו: הוא לא יצר אותו, אינו אחראי לו \
ואינו נותן לו תמיכה. המודל הוא של פרדס, ואינו בא מהחיפוש הסמנטי של זית.

The model is derived from neodictabert-bilingual-embed by Dicta: The Israel Center for Text Analysis, and released \
under CC BY 4.0; NOTICE.md travels with it.
"""


class Refused(Exception):
    """What stops the release; its message never holds the account's name."""


def weights_id(folder: Path, name: str) -> str:
    """pardes_semantic.model.model_id of a sentence-transformers folder called [name]."""
    h = hashlib.sha256()
    for f in sorted(folder.rglob("*.safetensors")):
        with f.open("rb") as r:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                h.update(chunk)
    return f"{name}@{h.hexdigest()[:7]}/{CLEANER_ID}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as r:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(model: str, folder: Path) -> dict:
    """The model's id, dimension and layers, after checking the folder holds what such a model has."""
    _, out_folder, _, _ = KINDS[model]
    if model == "student-onnx":
        manifest_path = folder / "pardes-model.json"
        if not manifest_path.is_file():
            raise Refused(f"{folder} has no pardes-model.json")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        for key, digest in (("onnx", "onnxSha256"), ("tokenizer", "tokenizerSha256")):
            f = folder / manifest[key]
            if not f.is_file() or sha256(f) != manifest[digest]:
                raise Refused(f"{f.name} is missing or does not match pardes-model.json")
        if not (folder / "NOTICE.md").is_file():
            raise Refused(f"{folder} has no NOTICE.md")
        return {"id": manifest["id"], "dimension": manifest["dimension"], "layers": None}
    if not (folder / "modules.json").is_file() or not any(folder.rglob("*.safetensors")):
        raise Refused(f"{folder} is not a sentence-transformers model (modules.json, *.safetensors)")
    config = json.loads((folder / "config.json").read_text("utf-8"))
    dimension = config.get("hidden_size")
    for module in json.loads((folder / "modules.json").read_text("utf-8")):
        if module.get("type", "").endswith(".Dense"):
            dimension = json.loads((folder / module["path"] / "config.json").read_text("utf-8"))["out_features"]
    layers = next((config[k] for k in ("num_hidden_layers", "num_layers", "n_layers") if k in config), None)
    return {"id": weights_id(folder, out_folder), "dimension": dimension, "layers": layers}


def owner_patterns(owner: str) -> tuple[re.Pattern, re.Pattern]:
    """Where a file names the account: as a path's part or before a kernel's or dataset's name (rewritten in a text
    file), and anywhere as a word (in the file names and the note)."""
    o = re.escape(owner.encode("utf-8"))
    in_path = re.compile(rb"(?<=[/\\])" + o + rb"(?=[/\\])|(?<![A-Za-z0-9_.-])" + o + rb"(?=/pardes-semantic)",
                         re.IGNORECASE)
    word = re.compile(r"(?<![A-Za-z0-9])" + re.escape(owner) + r"(?![A-Za-z0-9])", re.IGNORECASE)
    return in_path, word


def scrub(folder: Path, owner: str) -> int:
    """Rewrites to <owner> what names [owner] in the folder's text files; how many files it rewrote. Refused when a
    binary file or a file's name holds it."""
    if not owner:
        return 0
    in_path, word = owner_patterns(owner)
    rewritten = 0
    for f in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = f.relative_to(folder).as_posix()
        if word.search(rel):
            raise Refused(f"a file's name in {folder.name} names the Kaggle account")
        data = f.read_bytes()
        if not in_path.search(data):
            continue
        if f.suffix.lower() not in TEXT and f.name != "README":
            raise Refused(f"{rel} names the Kaggle account and is not a text file: not released")
        f.write_bytes(in_path.sub(OWNER_TEXT.encode(), data))
        rewritten += 1
    return rewritten


def pack(folder: Path, inner: str, extra: dict[str, Path], out: Path, tag: str,
         part_bytes: int = PART_BYTES) -> tuple[list[Path], str, int]:
    """[folder] zipped as [inner]/ with [extra] beside its files: the release's files (the zip, or its parts), the
    whole zip's sha256 and size."""
    out.mkdir(parents=True, exist_ok=True)
    whole = out / f"{tag}.zip"
    with zipfile.ZipFile(whole, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        names = set()
        for f in sorted(p for p in folder.rglob("*") if p.is_file()):
            name = f.relative_to(folder).as_posix()
            names.add(name)
            z.write(f, f"{inner}/{name}")
        for name, src in extra.items():
            if name not in names:
                z.write(src, f"{inner}/{name}")
    digest, size = sha256(whole), whole.stat().st_size
    if size <= part_bytes:
        return [whole], digest, size
    parts = []
    with whole.open("rb") as r:
        for i in range(1, 1000):
            chunk = r.read(part_bytes)
            if not chunk:
                break
            part = out / f"{tag}.zip.{i:03d}"
            part.write_bytes(chunk)
            parts.append(part)
    whole.unlink()
    return parts, digest, size


def human(size: int) -> str:
    return f"{size / 1e9:.2f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"


def notes(model: str, info: dict, files: list[Path], digest: str, size: int, *, version: str, note: str,
          repo_url: str, commit: str, today: str) -> str:
    stage, _, inner, _ = KINDS[model]
    onnx = model == "student-onnx"
    shape = f"{info['dimension']} ממדים" + (f", {info['layers']} שכבות" if info["layers"] else "")
    source = f"שלב `{stage}` ב־Kaggle, " + (f"גרסה {version}" if version else f"הפלט האחרון שלו ב־{today}")
    code = f"{repo_url}/blob/{commit}"
    lines = []
    if note:
        lines += [note, ""]
    lines += [
        DESCRIPTION[model],
        "",
        f"- **מזהה:** `{info['id']}`. " + ("זה גם המזהה של התלמיד שממנו יוצא, ושל אינדקס הווקטורים שהוטמע בו."
                                          if onnx else "אותו מזהה נמצא ב־`pardes-model.json` של מודל ה־ONNX שמיוצא "
                                          "ממנו, ובאינדקס הווקטורים שהוטמע בו."),
        f"- **צורה:** {shape}.",
        f"- **מקור:** {source}. " + (f"הקוד שייצא אותו: [`export_onnx.py`]({code}/export_onnx.py)" if onnx else
                                    f"הקוד שאימן אותו: [`kaggle_run.py`]({code}/kaggle_run.py)")
        + f", והתוכנית: [`TRAINING_PLAN.md`]({code}/TRAINING_PLAN.md).",
        "",
        (USE_ONNX if onnx else USE_TORCH).format(inner=inner),
        "**קבצים**",
    ]
    tag = files[0].name.split(".zip")[0]
    if len(files) == 1:
        lines.append(f"- `{files[0].name}`: התיקייה `{inner}/`, {human(size)}.")
    else:
        first, last = files[0].name, files[-1].name
        plus = "+".join(f.name for f in files)
        lines.append(f"- `{first}` עד `{last}`: התיקייה `{inner}/` ב־zip אחד של {human(size)}, בחלקים מתחת ל־2 GiB לקובץ "
                     f"של GitHub. מחברים אותם לפני הפריסה: `cat {tag}.zip.0* > {tag}.zip` בלינוקס ובמק, "
                     f"`copy /b {plus} {tag}.zip` ב־Windows, או פותחים את הראשון ב־7-Zip. ה־sha256 של ה־zip המחובר: "
                     f"`{digest}`.")
    lines += ["- `SHA256SUMS`: ה־sha256 של כל קובץ כאן (`sha256sum -c SHA256SUMS`).", "", LICENCE.format(inner=inner)]
    return "\n".join(lines)


def release(args: argparse.Namespace) -> None:
    owner = os.environ.get("KAGGLE_OWNER", "")
    if owner and any(owner_patterns(owner)[1].search(s) for s in (args.note, args.tag)):
        raise Refused("the note or the tag names the Kaggle account")
    stage, _, inner, name = KINDS[args.model]
    if not args.folder.is_dir():
        raise Refused(f"{args.folder} is not there: the output of {stage} has no such folder")
    rewritten = scrub(args.folder, owner)
    if rewritten:
        print(f"{rewritten} files named the Kaggle account in a path: rewritten to {OWNER_TEXT}")
    info = describe(args.model, args.folder)
    short = info["id"].split("/")[0]
    tag = args.tag or f"{args.model}-{short.split('@')[1]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise Refused(f"{tag} is not a tag")
    extra = {"NOTICE.md": HERE / "NOTICE.md"}
    files, digest, size = pack(args.folder, inner, extra, args.out, tag, args.part_bytes)
    (args.out / "SHA256SUMS").write_text("".join(f"{sha256(f)}  {f.name}\n" for f in files), "utf-8")
    args.meta.mkdir(parents=True, exist_ok=True)
    (args.meta / "tag.txt").write_text(tag, "utf-8")
    (args.meta / "title.txt").write_text(f"{name} · {short}", "utf-8")
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    text = notes(args.model, info, files, digest, size, version=args.version, note=args.note,
                 repo_url=args.repo_url, commit=args.commit, today=today)
    (args.meta / "notes.md").write_text(text, "utf-8")
    print(f"{tag}: {info['id']}, {human(size)} in {len(files)} file(s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-of", choices=KINDS, help="print the model's Kaggle stage and output folder, and stop")
    ap.add_argument("--model", choices=KINDS)
    ap.add_argument("--folder", type=Path, help="the model's folder as the stage's output holds it")
    ap.add_argument("--out", type=Path, default=Path("release"), help="the release's files")
    ap.add_argument("--meta", type=Path, default=Path("meta"), help="its tag, title and notes")
    ap.add_argument("--version", default="", help="the stage's version on Kaggle, for the notes (empty: the latest)")
    ap.add_argument("--note", default="", help="a line at the top of the notes")
    ap.add_argument("--tag", default="", help="the tag (default: <model>-<hash of its weights>)")
    ap.add_argument("--repo-url", default="https://github.com/efrul/pardes-semantic")
    ap.add_argument("--commit", default="main", help="the commit the notes link the code at")
    ap.add_argument("--part-bytes", type=int, default=PART_BYTES)
    args = ap.parse_args()
    if args.source_of:
        stage, folder, _, _ = KINDS[args.source_of]
        print(stage, folder)
        return
    if not args.model or not args.folder:
        ap.error("--model and --folder are needed")
    try:
        release(args)
    except Refused as e:
        sys.exit(f"Not released: {e}")


if __name__ == "__main__":
    main()

"""release_model.py on small made-up model folders: the id the app checks, the shape, the account kept out, the zip
and its parts, the notes.

    python test_release_model.py      (or pytest)
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_model as r  # noqa: E402

OWNER = "someone"


def st_model(root: Path, name: str = "student") -> Path:
    """A sentence-transformers folder as distill_model.py saves one: a 7-layer model projected to 384 dims."""
    d = root / name
    (d / "1_Pooling").mkdir(parents=True)
    (d / "2_Dense").mkdir()
    (d / "model.safetensors").write_bytes(os.urandom(3000))
    (d / "2_Dense" / "model.safetensors").write_bytes(os.urandom(500))
    (d / "config.json").write_text(json.dumps({"hidden_size": 768, "num_hidden_layers": 7}), "utf-8")
    (d / "2_Dense" / "config.json").write_text(json.dumps({"in_features": 768, "out_features": 384}), "utf-8")
    (d / "modules.json").write_text(json.dumps([
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"}]), "utf-8")
    return d


def model_id(source: Path) -> str:
    """pardes_semantic.model.model_id, as it is (that module needs torch)."""
    h = hashlib.sha256()
    for f in sorted(source.rglob("*.safetensors")):
        h.update(f.read_bytes())
    return f"{source.name}@{h.hexdigest()[:7]}/{r.CLEANER_ID}"


def test_the_id_is_the_apps():
    with tempfile.TemporaryDirectory() as t:
        d = st_model(Path(t))
        info = r.describe("student", d)
        assert info["id"] == model_id(d)
        assert (info["dimension"], info["layers"]) == (384, 7)


def test_onnx_model_checked_against_its_manifest():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "model-student"
        d.mkdir()
        (d / "model.int8.onnx").write_bytes(b"onnx")
        (d / "tokenizer.json").write_text("{}", "utf-8")
        (d / "NOTICE.md").write_text("notice", "utf-8")
        manifest = {"id": "student@695cbe2/pardes-clean-1", "dimension": 384, "onnx": "model.int8.onnx",
                    "onnxSha256": r.sha256(d / "model.int8.onnx"), "tokenizer": "tokenizer.json",
                    "tokenizerSha256": r.sha256(d / "tokenizer.json")}
        (d / "pardes-model.json").write_text(json.dumps(manifest), "utf-8")
        assert r.describe("student-onnx", d)["id"] == "student@695cbe2/pardes-clean-1"
        (d / "model.int8.onnx").write_bytes(b"other")
        try:
            r.describe("student-onnx", d)
            raise AssertionError("a changed ONNX file was released")
        except r.Refused:
            pass


def test_the_account_rewritten_in_paths_only():
    with tempfile.TemporaryDirectory() as t:
        d = st_model(Path(t))
        config = {"_name_or_path": f"/kaggle/input/notebooks/{OWNER}/pardes-semantic-train/teacher",
                  "base": "Someone/pardes-semantic-train"}
        (d / "config_sentence_transformers.json").write_text(json.dumps(config), "utf-8")
        # A word of the vocabulary that is the account's name is not a path: kept
        (d / "tokenizer.json").write_text(json.dumps({"vocab": {OWNER: 5, "x": 6}}), "utf-8")
        assert r.scrub(d, OWNER) == 1
        rewritten = json.loads((d / "config_sentence_transformers.json").read_text("utf-8"))
        assert rewritten == {"_name_or_path": "/kaggle/input/notebooks/<owner>/pardes-semantic-train/teacher",
                             "base": "<owner>/pardes-semantic-train"}
        assert OWNER in (d / "tokenizer.json").read_text("utf-8")


def test_the_account_in_a_binary_file_or_a_name_refused():
    for where in ("binary", "name"):
        with tempfile.TemporaryDirectory() as t:
            d = st_model(Path(t))
            if where == "binary":
                (d / "model.safetensors").write_bytes(b"\x00\x01/kaggle/input/" + OWNER.encode() + b"/x\x00")
            else:
                (d / f"{OWNER}.txt").write_text("x", "utf-8")
            try:
                r.scrub(d, OWNER)
                raise AssertionError(f"the account's name in a {where} was released")
            except r.Refused as e:
                assert OWNER not in str(e)


def test_parts_join_to_the_zip():
    with tempfile.TemporaryDirectory() as t:
        d = st_model(Path(t))
        notice = Path(t) / "NOTICE.md"
        notice.write_text("notice", "utf-8")
        files, digest, size = r.pack(d, "student", {"NOTICE.md": notice}, Path(t) / "out", "student-abc",
                                     part_bytes=1000)
        assert len(files) > 1 and [f.name for f in files][:2] == ["student-abc.zip.001", "student-abc.zip.002"]
        joined = Path(t) / "joined.zip"
        joined.write_bytes(b"".join(f.read_bytes() for f in files))
        assert r.sha256(joined) == digest and joined.stat().st_size == size
        with zipfile.ZipFile(joined) as z:
            names = set(z.namelist())
        assert {"student/model.safetensors", "student/2_Dense/config.json", "student/NOTICE.md"} <= names


def test_release_and_its_notes():
    with tempfile.TemporaryDirectory() as t:
        d = st_model(Path(t))
        os.environ["KAGGLE_OWNER"] = OWNER
        try:
            for note in (f"of {OWNER}'s run", ""):
                args = argparse.Namespace(model="student", folder=d, out=Path(t) / "release", meta=Path(t) / "meta",
                                          version="", note=note, tag="", repo_url="https://github.com/x/y",
                                          commit="abc123", part_bytes=r.PART_BYTES)
                if note:
                    try:
                        r.release(args)
                        raise AssertionError("a note naming the account was released")
                    except r.Refused:
                        continue
                r.release(args)
        finally:
            del os.environ["KAGGLE_OWNER"]
        short = model_id(d).split("/")[0]
        assert (Path(t) / "meta" / "tag.txt").read_text("utf-8") == f"student-{short.split('@')[1]}"
        assert (Path(t) / "meta" / "title.txt").read_text("utf-8") == f"התלמיד · {short}"
        notes = (Path(t) / "meta" / "notes.md").read_text("utf-8")
        assert model_id(d) in notes and "384 ממדים, 7 שכבות" in notes and "CC BY 4.0" in notes
        assert "https://github.com/x/y/blob/abc123/kaggle_run.py" in notes and "{" not in notes.split("```")[0]
        sums = (Path(t) / "release" / "SHA256SUMS").read_text("utf-8").split()
        assert sums[1] == f"student-{short.split('@')[1]}.zip"


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)

"""collect.py's write_daf against packages whose tractate is not a plain name, which must not name a file elsewhere:

    python feedback/test_collect.py      (or pytest)
"""

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import write_daf  # noqa: E402


def mark():
    return {"amud": "2a", "c": "gemara", "i": 3, "t": 0, "ok": True, "at": "2026-11-05T10:00:00.123456789Z"}


def test_only_a_plain_tractate_name_is_written():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "out"
        out.mkdir()
        daf = [
            ("a", {"tractate": "Berakhot", "marks": [mark()]}),
            ("b", {"tractate": "../../escaped", "marks": [mark()]}),
            ("c", {"tractate": str(root / "absolute"), "marks": [mark()]}),
        ]
        with redirect_stdout(io.StringIO()) as printed:
            write_daf(daf, out)
        assert sorted(p.name for p in (out / "daf").iterdir()) == ["Berakhot.json"]
        assert not (root / "escaped.json").exists()
        assert not (root / "absolute.json").exists()
        assert json.loads((out / "daf" / "Berakhot.json").read_text("utf-8"))["tractate"] == "Berakhot"
        assert "2 reviews left out" in printed.getvalue()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok {name}")

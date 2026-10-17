import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_the_paper_names_the_membrane(tmp_path: Path) -> None:
    out = tmp_path / "angularai.pdf"
    sys.argv = ["build_paper.py", str(out)]
    runpy.run_path(str(ROOT / "docs" / "build_paper.py"), run_name="__main__")
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    assert b"AngularAI" in data
    assert b"membrane" in data
    assert b"notebook" in data
    assert b"passes.jsonl" in data
    pages = data.count(b"/Type /Page") - data.count(b"/Type /Pages")
    assert 2 <= pages <= 4

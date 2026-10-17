from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_the_page_states_the_cortex() -> None:
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    style = (ROOT / "docs" / "site.css").read_text(encoding="utf-8")
    assert "A Cortex In Front Of The Model You Already Call" in page
    assert "angularai.pdf" in page
    assert "uv tool install git+https://github.com/baiehclaca/AngularAI.git" in page
    assert "The notebook is blue." in page
    assert "What shade is it?" in page
    assert "site.css" in page
    assert ".sky" in style
    lowered = page.lower()
    assert "nitra" not in lowered
    assert "typesafe" not in lowered
    assert "jev" not in lowered

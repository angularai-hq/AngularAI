import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_the_page_states_the_cortex() -> None:
    docs = ROOT / "docs"
    page = (docs / "index.html").read_text(encoding="utf-8")
    platform = (docs / "platform" / "index.html").read_text(encoding="utf-8")
    mcp = (docs / "mcp" / "index.html").read_text(encoding="utf-8")
    assert "Angular AI" in page
    assert "Thameleon" in page
    assert "angularai.pdf" in page
    assert "https://angularai-hq.github.io/AngularAI/install.sh | sh" in page
    assert "packages/angularai-mcp" in page
    assert "cortexes" in mcp
    script = (docs / "install.sh").read_text(encoding="utf-8")
    assert "git+https://github.com/angularai-hq/AngularAI.git" in script
    assert (docs / "angularai.pdf").is_file()
    assert (docs / "logo.png").is_file()
    assert (docs / ".nojekyll").is_file()
    assert (docs / "404.html").is_file()
    assert "The notebook is blue." in platform
    assert "What shade is it?" in platform
    for html in docs.rglob("*.html"):
        lowered = html.read_text(encoding="utf-8").lower()
        for word in ("nitra", "typesafe", "jev"):
            assert word not in lowered, (html, word)


def test_the_installer_prints_a_plan_without_installing() -> None:
    env = os.environ.copy()
    env["ANGULARAI_INSTALL_PLAN"] = "1"
    result = subprocess.run(
        ["sh", str(ROOT / "docs" / "install.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "angularai" in result.stdout
    assert "git+https://github.com/angularai-hq/AngularAI.git" in result.stdout
    assert "sk-" not in result.stdout + result.stderr

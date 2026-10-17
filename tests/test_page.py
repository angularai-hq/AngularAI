import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_the_page_states_the_cortex() -> None:
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    style = (ROOT / "docs" / "site.css").read_text(encoding="utf-8")
    assert "A Cortex In Front Of The Model You Already Call" in page
    assert "angularai.pdf" in page
    assert "install.sh" in page
    script = (ROOT / "docs" / "install.sh").read_text(encoding="utf-8")
    assert "git+https://github.com/angularai-hq/AngularAI.git" in script
    assert "logo.png" in page
    assert "The notebook is blue." in page
    assert "What shade is it?" in page
    assert "site.css" in page
    assert ".sky" in style
    lowered = page.lower()
    assert "nitra" not in lowered
    assert "typesafe" not in lowered
    assert "jev" not in lowered


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

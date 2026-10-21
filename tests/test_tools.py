from pathlib import Path

from angularai.models import Action
from angularai.tools import execute


def test_a_path_outside_the_directory_is_not_read(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("the vault code is cedar-plank-9\n")
    result = execute(root, Action("read", path="../outside.txt"))
    assert "leaves the directory" in result
    assert "cedar-plank-9" not in result
    assert outside.read_text(encoding="utf-8").startswith("the vault")


def test_a_symlink_cannot_leave_the_directory(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("hidden-outside\n")
    (root / "link.txt").symlink_to(outside)
    result = execute(root, Action("read", path="link.txt"))
    assert "hidden-outside" not in result


def test_a_secret_in_a_file_is_redacted(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    (root / "key.txt").write_text("token sk-test-secret-value-xyz\n")
    result = execute(root, Action("read", path="key.txt"))
    assert "sk-test-secret-value-xyz" not in result
    assert "[redacted]" in result


def test_edit_replaces_one_span(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    note = root / "note.txt"
    note.write_text("color = green\n", encoding="utf-8")
    result = execute(root, Action("edit", path="note.txt", old="green", new="blue"))
    assert "Edited" in result
    assert note.read_text(encoding="utf-8") == "color = blue\n"
    ambiguous = execute(root, Action("edit", path="note.txt", old="o", new="a"))
    assert "times" in ambiguous
    assert note.read_text(encoding="utf-8") == "color = blue\n"


def test_search_stays_inside_and_run_refuses_a_shell(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    (root / "seen.txt").write_text("The notebook is blue.\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("The notebook is elsewhere.\n", encoding="utf-8")
    found = execute(root, Action("search", path=".", query="notebook"))
    assert "seen.txt" in found
    assert "elsewhere" not in found
    off = execute(root, Action("run", command="ls"))
    assert "run is off" in off
    refused = execute(root, Action("run", command="bash -c 'echo hi'"), allow_run=True)
    assert "Refused bash" in refused
    left = execute(root, Action("run", command="rm -rf /"), allow_run=True)
    assert "leaves the directory" in left


def test_a_missed_edit_quotes_the_closest_line(tmp_path: Path) -> None:
    (tmp_path / "z.py").write_text('def z(x):\n    """TODO: describe."""\n    return x\n')
    missed = execute(tmp_path, Action("edit", path="z.py", old='"""TODO: describe.""" ', new="x"))
    assert missed.startswith("That text is not in the file. Line 2 is close: ")
    assert '"    \\"\\"\\"TODO: describe.\\"\\"\\""' in missed
    far = execute(tmp_path, Action("edit", path="z.py", old="completely unrelated", new="x"))
    assert far == "That text is not in the file."
    blank = execute(tmp_path, Action("edit", path="z.py", old="\n\n", new="x"))
    assert blank == "That text is not in the file."

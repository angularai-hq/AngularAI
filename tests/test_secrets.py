from pathlib import Path

import pytest

from angularai.agent import run_agent
from angularai.configure import add_lock
from angularai.lock import HELD, ValueLock, normalize
from angularai.models import Action, Integration
from angularai.pilot import pass_turn
from angularai.secrets import (
    contains_secret,
    environment_secrets,
    looks_secret,
    redact,
    secret_name,
)
from angularai.store import Store, init_cortex, open_store
from tests.scripted import Scripted

# Fake values. Each has the shape of a real key and no other meaning.
SECRETS = [
    ("openai", "sk-proj-Ab3dEf6hIj9kLm2nOp5qRs8tUv"),
    ("openai legacy", "sk-abcdefghijklmnopqrstuvwxyz012345"),
    ("anthropic", "sk-ant-api03-Ab3dEf6hIj9kLm2nOp5qRs8tUv"),
    (
        "github fine-grained",
        "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz0123456789ABC",
    ),
    ("github classic", "ghp_abcdefghijklmnopqrstuvwxyz0123456789"),
    ("github oauth", "gho_abcdefghijklmnopqrstuvwxyz0123456789"),
    ("github server", "ghs_abcdefghijklmnopqrstuvwxyz0123456789"),
    ("github user", "ghu_abcdefghijklmnopqrstuvwxyz0123456789"),
    ("github refresh", "ghr_abcdefghijklmnopqrstuvwxyz0123456789"),
    ("stripe live", "sk_live_51Habcdefghijklmnop"),
    ("stripe restricted", "rk_live_51Habcdefghijklmnop"),
    ("google", "AIzaSyA-abcdefghijklmnopqrstuvwxyz12345"),
    ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijklmnop"),
    ("gitlab", "glpat-abcdefghijklmnopqrst"),
    ("huggingface", "hf_abcdefghijklmnopqrstuvwxyzABCDEF"),
    ("slack bot", "xoxb-1234567890-abcdefghij"),
    ("slack user", "xoxp-1234567890-abcdefghij"),
    ("slack app", "xoxa-1234567890-abcdefghij"),
    ("aws", "AKIAIOSFODNN7EXAMPLE"),
    ("pem header", "-----BEGIN OPENSSH PRIVATE KEY-----"),
    ("password", "password=hunter2hunter2"),
    ("token yaml", "token: abc123def456"),
    ("api key json", '"api_key": "Zq81mPx0Lk"'),
    ("entropy", "Xk9mQ2vL7pR4tY8wN3bF6hJ1sD5gA0cZ"),
]

ORDINARY = [
    ("desk", "Put it on the desk-organizer shelf"),
    ("ask", "ask-anything mode"),
    ("risk", "the risk-assessment doc"),
    ("tusk", "tusk-elephants roam"),
    ("sk alone", "the sk- prefix"),
    ("short sk", "sk-short"),
    ("uuid", "id 3f2b8c1e-9d4a-4e7b-8f6c-2a1d5e9b7c40 is the row"),
    ("git sha", "commit 1c435be48d3f4e95b32b1e777d74f929b1707f18"),
    ("sha256", "sum 2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"),
    ("camel", "getUserAccountSettingsFromDatabase2024"),
    ("path", "src/components/UserProfileSettings/index.tsx"),
    ("snake", "test_a_long_reply_arrives_in_full_with_style"),
    ("wheel", "angularai-0.1.0-py3-none-any.whl-2026-10-17"),
    ("code read", "password = os.environ['DB_PASSWORD']"),
    ("code call", "token = get_token()"),
    ("placeholder", "password=********"),
    ("template", "token: <your token>"),
    ("prose", "My password: forgotten"),
    ("null", "secret=None"),
    ("words", "I have a blue notebook."),
    (
        "macos temp",
        "/private/var/folders/xd/1b2c3d4e5f6g7h8j9k0l1m2n3p000gn/T/pytest-of-runner/pytest-0/work",
    ),
    ("relative path", "./build/Xk9mQ2vL/output/Release/AngularAI-Setup-2026.dmg"),
]


@pytest.mark.parametrize(("name", "text"), SECRETS, ids=[name for name, _ in SECRETS])
def test_a_secret_is_seen_and_redacted(name: str, text: str) -> None:
    assert looks_secret(f"note {text} end"), name
    redacted = redact(f"note {text} end", ())
    assert "[redacted]" in redacted
    assert redacted.startswith("note ")
    assert redacted.endswith(" end")


@pytest.mark.parametrize(("name", "text"), ORDINARY, ids=[name for name, _ in ORDINARY])
def test_ordinary_text_is_not_a_secret(name: str, text: str) -> None:
    assert not looks_secret(text), name
    assert redact(text, ()) == text


def test_a_random_segment_in_a_path_is_redacted_alone() -> None:
    path = "/tmp/Xk9mQ2vL7pR4tY8wN3bF6hJ1sD5gA0cZ/out.txt"
    assert looks_secret(path)
    assert redact(path, ()) == "/tmp/[redacted]/out.txt"


def test_an_assignment_keeps_its_name() -> None:
    assert redact("password=hunter2hunter2 next", ()) == "password=[redacted] next"


def test_a_pem_block_is_redacted_whole() -> None:
    block = (
        "before\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\nabc\n"
        "-----END RSA PRIVATE KEY-----\nafter"
    )
    assert redact(block, ()) == "before\n[redacted]\nafter"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("OPENAI_API_KEY", True),
        ("GITHUB_TOKEN_BAIEHCLACA", True),
        ("AWS_SECRET_ACCESS_KEY", True),
        ("DB_PASSWORD", True),
        ("SMTP_PASSWD", True),
        ("GOOGLE_APPLICATION_CREDENTIALS", True),
        ("SLACK_APIKEY", True),
        ("HF_AUTHTOKEN", True),
        ("PATH", False),
        ("HOME", False),
        ("KEYBOARD_LAYOUT", False),
        ("TOKENIZERS_PARALLELISM", False),
        ("MONKEY", False),
    ],
)
def test_a_secret_name_is_recognized(name: str, expected: bool) -> None:
    assert secret_name(name) is expected


def test_environment_values_are_secrets_by_name() -> None:
    environ = {
        "GITHUB_TOKEN_X": "plainwordvalue",
        "DB_PASSWORD": "pa55wd",
        "ANGULARAI_API_KEY": "local",
        "FEATURE_TOKEN": "true",
        "PIN_SECRET": "123456",
        "EDITOR": "vim-is-a-long-value",
    }
    found = environment_secrets(environ)
    assert found == ("plainwordvalue", "pa55wd")
    assert redact("the plainwordvalue here", found) == "the [redacted] here"
    assert redact("pa55wd and xpa55wdx", found) == "[redacted] and xpa55wdx"
    assert contains_secret("use pa55wd", found)
    assert not contains_secret("a local variable", environment_secrets(environ))


@pytest.mark.parametrize(
    "reply",
    [
        "See the dosing chart.",
        "See the DOSING  CHART.",
        "See the dosing\u200bchart.".replace("\u200b", "\u200b "),
        "See the do\u200bsing chart.",
        "See the \uff44\uff4f\uff53\uff49\uff4e\uff47 chart.",
        "See the dosing\n\tchart.",
    ],
)
def test_the_lock_normalizes_before_it_matches(reply: str) -> None:
    verdict = ValueLock().judge(reply, ("dosing chart",), ())
    assert verdict.allowed is False
    assert verdict.text == HELD


def test_normalize() -> None:
    assert normalize("a\u200b b\u00a0\u00a0c \uff21") == "a b c A"


def test_a_secret_split_by_a_zero_width_space_is_held() -> None:
    split = "sk-proj-Ab3dEf6hIj\u200b9kLm2nOp5qRs8tUv"
    assert ValueLock().judge(f"key {split}", (), ()).allowed is False


def _store(tmp_path: Path) -> Store:
    init_cortex(tmp_path / "cortex")
    return open_store(tmp_path / "cortex")


def test_a_forbidden_phrase_in_an_act_holds_it(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    store = _store(tmp_path)
    add_lock(store, "dosing chart")
    act = Action("write", path="note.txt", text="The dosing chart is attached.")
    result = run_agent(store, Scripted(Integration("Writing.", (), (), act)), "Write.", root)
    assert result.stop == "held"
    assert not (root / "note.txt").exists()


def test_a_secret_in_an_act_holds_it(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    store = _store(tmp_path)
    act = Action("write", path="env.txt", text="OPENAI=sk-proj-Ab3dEf6hIj9kLm2nOp5qRs8tUv")
    result = run_agent(store, Scripted(Integration("Writing.", (), (), act)), "Write.", root)
    assert result.stop == "held"
    assert not (root / "env.txt").exists()
    log = (tmp_path / "cortex" / "passes.jsonl").read_text(encoding="utf-8")
    assert "Ab3dEf6hIj" not in log


def test_the_log_keeps_the_tool_not_its_output(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    (root / "seen.txt").write_text("The vault code is cedar-plank-9.\n", encoding="utf-8")
    store = _store(tmp_path)
    scripted = Scripted(
        script=(
            Integration("Reading.", (), (), Action("read", path="seen.txt")),
            Integration("Done.", (), (), None),
        )
    )
    run_agent(store, scripted, "Read the note.", root)
    assert "cedar-plank-9" in scripted.seen[1].user
    log = (tmp_path / "cortex" / "passes.jsonl").read_text(encoding="utf-8")
    assert "cedar-plank-9" not in log
    assert "read  seen.txt" in log
    users = [record.user for record in store.passes()]
    assert users == ["TASK\nRead the note.", "TASK\nRead the note.\n\nSTEP 2 of 8\nread  seen.txt"]


def test_say_still_keeps_the_input(tmp_path: Path) -> None:
    store = _store(tmp_path)
    pass_turn(store, Scripted(Integration("Noted.", (), ())), "I have a blue notebook.")
    assert store.passes()[-1].user == "I have a blue notebook."

"""Saved models and keys: the keyring, the 0600 fallback file, and which model wins."""

import json
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path

import keyring
import pytest
from keyring.backends import fail, null

from angularai.errors import ConfigError, InputRejected, KeyMissing
from angularai.models import Config
from angularai.profiles import (
    KEYS_FILE,
    MODELS_FILE,
    SERVICE,
    Backend,
    Book,
    Profile,
    Vault,
    add_profile,
    anything_configured,
    check_key,
    check_model,
    default_backend,
    effective_config,
    free_name,
    key_source,
    load_book,
    mask,
    remove_profile,
    storage_words,
    use_profile,
)
from angularai.store import default_config
from angularai.substrate import _openai_headers, build_substrate, secret_values
from tests.setupfakes import KEY, BrokenKeyring, FakeKeyring, SilentKeyring


def env_for(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {"ANGULARAI_HOME": str(tmp_path / "home"), **extra}


def profile(name: str = "openai", provider: str = "openai") -> Profile:
    if provider == "compatible":
        return Profile(name, "compatible", "llama3.2", "http://127.0.0.1:11434/v1", "none", "")
    if provider == "anthropic":
        return Profile(name, "anthropic", "claude-x", None, "none", "")
    return Profile(name, "openai", "gpt-x", None, "none", "")


def none_backend() -> Backend | None:
    return None


def fixed(backend: Backend) -> Callable[[], Backend | None]:
    return lambda: backend


def all_text(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in root.rglob("*")
        if path.is_file()
    )


# --- storage ----------------------------------------------------------------


def test_the_key_goes_to_the_keyring_and_not_to_disk(
    tmp_path: Path, fake_keyring_and_home: FakeKeyring
) -> None:
    environ = env_for(tmp_path)
    saved = add_profile(environ, Vault(environ), profile(), KEY, make_active=True)
    assert saved.storage == "keyring"
    assert saved.hint == "sk-…a1b2"
    account = f"{tmp_path / 'home'}::openai"
    assert fake_keyring_and_home.store[(SERVICE, account)] == KEY
    assert not (tmp_path / "home" / KEYS_FILE).exists()
    assert KEY not in all_text(tmp_path)
    book = load_book(environ)
    assert book.active == "openai"
    assert book.get("openai") == saved
    assert Vault(environ).load("openai", "keyring") == KEY
    assert storage_words("keyring", environ) == "the OS keyring"


@pytest.mark.parametrize("backend", [none_backend, fixed(BrokenKeyring()), fixed(SilentKeyring())])
def test_without_a_usable_keyring_the_key_goes_to_a_0600_file(
    tmp_path: Path, backend: Callable[[], Backend | None]
) -> None:
    environ = env_for(tmp_path)
    vault = Vault(environ, backend)
    saved = add_profile(environ, vault, profile(), KEY, make_active=True)
    assert saved.storage == "file"
    path = tmp_path / "home" / KEYS_FILE
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "home").stat().st_mode) == 0o700
    assert json.loads(path.read_text())["keys"] == {"openai": KEY}
    assert KEY not in (tmp_path / "home" / MODELS_FILE).read_text()
    assert vault.load("openai", "file") == KEY
    assert "0600" in storage_words("file", environ)
    assert repr(vault) == "Vault()"


def test_the_key_file_is_0600_from_its_first_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ = env_for(tmp_path)
    modes: list[int] = []
    real = tempfile.mkstemp

    def watching(*args: object, **kwargs: object) -> tuple[int, str]:
        descriptor, name = real(*args, **kwargs)  # type: ignore[call-overload]
        modes.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
        return descriptor, name

    monkeypatch.setattr(tempfile, "mkstemp", watching)
    old = os.umask(0)
    try:
        Vault(environ, none_backend).save("openai", KEY)
    finally:
        os.umask(old)
    assert modes and all(mode == 0o600 for mode in modes)
    assert stat.S_IMODE((tmp_path / "home" / KEYS_FILE).stat().st_mode) == 0o600


def test_a_failed_write_leaves_the_old_key_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ = env_for(tmp_path)
    vault = Vault(environ, none_backend)
    vault.save("first", "sk-first-000000000000000000")
    before = (tmp_path / "home" / KEYS_FILE).read_bytes()

    def boom(source: object, target: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        vault.save("second", KEY)
    monkeypatch.undo()
    assert (tmp_path / "home" / KEYS_FILE).read_bytes() == before
    leftovers = [p.name for p in (tmp_path / "home").iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_a_key_file_others_can_read_is_refused(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    vault = Vault(environ, none_backend)
    vault.save("openai", KEY)
    path = tmp_path / "home" / KEYS_FILE
    path.chmod(0o644)
    with pytest.raises(ConfigError, match="chmod 600") as caught:
        vault.load("openai", "file")
    assert KEY not in str(caught.value)


def test_a_linked_or_broken_key_file_is_refused(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    (home / KEYS_FILE).symlink_to(target)
    with pytest.raises(ConfigError, match="regular file"):
        Vault(environ, none_backend).load("x", "file")
    (home / KEYS_FILE).unlink()
    (home / KEYS_FILE).write_text('{"version": 9}')
    (home / KEYS_FILE).chmod(0o600)
    with pytest.raises(ConfigError, match="not a key file"):
        Vault(environ, none_backend).load("x", "file")


def test_a_keyring_that_fails_on_read_names_the_fix_without_the_key(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    with pytest.raises(KeyMissing, match="Unlock") as caught:
        Vault(environ, fixed(BrokenKeyring())).load("openai", "keyring")
    assert KEY not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    with pytest.raises(KeyMissing, match="no keyring is usable"):
        Vault(environ, none_backend).load("openai", "keyring")
    assert Vault(environ).load("openai", "none") is None


def test_remove_deletes_the_keyring_entry(
    tmp_path: Path, fake_keyring_and_home: FakeKeyring
) -> None:
    environ = env_for(tmp_path)
    vault = Vault(environ)
    add_profile(environ, vault, profile(), KEY, make_active=True)
    add_profile(environ, vault, profile("other"), "sk-other-00000000000000zz", make_active=False)
    gone, was_active = remove_profile(environ, vault, "openai")
    assert gone.name == "openai" and was_active
    assert all(account[1].endswith("::other") for account in fake_keyring_and_home.store)
    book = load_book(environ)
    assert book.active is None
    assert [item.name for item in book.profiles] == ["other"]
    with pytest.raises(InputRejected, match="No saved model"):
        remove_profile(environ, vault, "openai")


def test_remove_deletes_the_file_entry(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    vault = Vault(environ, none_backend)
    add_profile(environ, vault, profile(), KEY, make_active=True)
    remove_profile(environ, vault, "openai")
    assert KEY not in (tmp_path / "home" / KEYS_FILE).read_text()


def test_replacing_moves_the_key_and_drops_the_old_entry(
    tmp_path: Path, fake_keyring_and_home: FakeKeyring
) -> None:
    environ = env_for(tmp_path)
    add_profile(environ, Vault(environ), profile(), KEY, make_active=True)
    assert fake_keyring_and_home.store
    silent = SilentKeyring()
    silent.store = fake_keyring_and_home.store
    other = "sk-other-0000000000000000zzzz"
    saved = add_profile(environ, Vault(environ, fixed(silent)), profile(), other, make_active=True)
    assert saved.storage == "file"
    assert fake_keyring_and_home.store == {}
    assert load_book(environ).get("openai") is not None
    # And back: the keyring takes it, the file forgets it.
    add_profile(environ, Vault(environ), profile(), KEY, make_active=True)
    assert KEY not in (tmp_path / "home" / KEYS_FILE).read_text()
    add_profile(environ, Vault(environ), profile(), None, make_active=True)
    assert fake_keyring_and_home.store == {}
    assert load_book(environ).get("openai") == profile()


def test_use_switches_and_refuses_unknown(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    vault = Vault(environ)
    add_profile(environ, vault, profile("a"), KEY, make_active=True)
    add_profile(environ, vault, profile("b", "compatible"), None, make_active=False)
    assert load_book(environ).active == "a"
    use_profile(environ, "b")
    assert load_book(environ).active == "b"
    with pytest.raises(InputRejected, match="No saved model"):
        use_profile(environ, "c")
    with pytest.raises(InputRejected, match="lowercase"):
        use_profile(environ, "Bad Name")


def test_free_name() -> None:
    book = Book("openai", (profile(), profile("openai-2")))
    assert free_name(book, "openai") == "openai-3"
    assert free_name(book, "anthropic") == "anthropic"


@pytest.mark.parametrize(
    ("payload", "words"),
    [
        ("not json", "not valid JSON"),
        ("[]", "not a JSON object"),
        ('{"version": 2, "models": []}', "version"),
        ('{"version": 1}', "no list"),
        ('{"version": 1, "models": [1]}', "not an object"),
        ('{"version": 1, "models": [], "active": "x"}', "active"),
        (
            '{"version": 1, "models": [{"name": "a", "provider": "x", "model": "m",'
            ' "key": "none"}]}',
            "unknown provider",
        ),
        (
            '{"version": 1, "models": [{"name": "a", "provider": "compatible", "model": "m",'
            ' "key": "none"}]}',
            "without base_url",
        ),
        (
            '{"version": 1, "models": [{"name": "a", "provider": "openai", "model": "m",'
            ' "key": "vault"}]}',
            "key storage",
        ),
        (
            '{"version": 1, "models": [{"name": "a", "provider": "openai", "model": "m",'
            ' "key": "none", "hint": "' + "x" * 40 + '"}]}',
            "hint",
        ),
        (
            '{"version": 1, "models": [{"name": "A B", "provider": "openai", "model": "m",'
            ' "key": "none"}]}',
            "lowercase",
        ),
        (
            '{"version": 1, "models": [{"name": "a", "provider": "openai", "model": "m",'
            ' "key": "none"}, {"name": "a", "provider": "openai", "model": "m",'
            ' "key": "none"}]}',
            "twice",
        ),
    ],
)
def test_a_damaged_models_file_is_refused_by_name(tmp_path: Path, payload: str, words: str) -> None:
    environ = env_for(tmp_path)
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / MODELS_FILE).write_text(payload)
    with pytest.raises(ConfigError, match=words):
        load_book(environ)
    assert anything_configured(environ)


def test_default_backend_skips_fail_and_null_keyrings() -> None:
    assert isinstance(default_backend(), FakeKeyring)
    keyring.set_keyring(fail.Keyring())  # type: ignore[no-untyped-call]
    assert default_backend() is None
    keyring.set_keyring(null.Keyring())  # type: ignore[no-untyped-call]
    assert default_backend() is None


def test_checks_never_echo_the_key() -> None:
    for bad in ("", "   ", "sk-has space-0000000000", "sk-ünïcode-000000000", "x" * 5000):
        with pytest.raises(InputRejected) as caught:
            check_key(bad)
        assert bad.strip() == "" or bad not in str(caught.value)
    assert check_key(f"  {KEY}\n") == KEY
    with pytest.raises(InputRejected):
        check_model("has space")
    assert check_model(" gpt-x ") == "gpt-x"


def test_mask_examples() -> None:
    assert mask(KEY) == "sk-…a1b2"
    assert mask("sk-ant-api03-000000000000wxyz") == "sk-ant-…wxyz"
    assert mask("sk-proj-0000000000000000abcd") == "sk-proj-…abcd"
    assert mask("abcdefghij") == "…ij"
    assert mask("short") == "…"


# --- precedence -------------------------------------------------------------


def wired() -> Config:
    base = default_config()
    return Config(
        "anthropic", "wired-model", None, base.workspace_budget, base.max_input_tokens, (), False
    )


def test_which_model_wins(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    vault = Vault(environ)
    add_profile(environ, vault, profile("main"), KEY, make_active=True)
    add_profile(environ, vault, profile("local", "compatible"), None, make_active=False)
    # Unwired cortex: the active saved model.
    config, chosen = effective_config(default_config(), environ)
    assert chosen is not None and chosen.name == "main"
    assert (config.substrate, config.model, config.json_mode) == ("openai", "gpt-x", True)
    # A wired cortex keeps its own wiring.
    config, chosen = effective_config(wired(), environ)
    assert chosen is None and config.model == "wired-model"
    # ANGULARAI_MODEL beats both.
    picked = {**environ, "ANGULARAI_MODEL": "local"}
    config, chosen = effective_config(wired(), picked)
    assert chosen is not None and chosen.name == "local"
    assert config.base_url == "http://127.0.0.1:11434/v1"
    with pytest.raises(ConfigError, match="no saved model"):
        effective_config(default_config(), {**environ, "ANGULARAI_MODEL": "gone"})
    with pytest.raises(ConfigError, match="must name"):
        effective_config(default_config(), {**environ, "ANGULARAI_MODEL": "Not A Name"})
    # Nothing saved, nothing wired: nothing.
    empty = env_for(tmp_path / "empty")
    assert effective_config(default_config(), empty) == (default_config(), None)


def test_the_environment_key_beats_the_saved_key(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    add_profile(environ, Vault(environ), profile(), KEY, make_active=True)
    substrate = build_substrate(default_config(), environ)
    assert substrate._api_key == KEY  # type: ignore[attr-defined]
    other = "sk-environment-000000000000ffff"
    substrate = build_substrate(default_config(), {**environ, "OPENAI_API_KEY": other})
    assert substrate._api_key == other  # type: ignore[attr-defined]
    assert key_source("openai", load_book(environ).get("openai"), environ) == (
        "sk-…a1b2 in the OS keyring"
    )
    assert key_source("openai", None, {"OPENAI_API_KEY": other}) == "OPENAI_API_KEY (environment)"
    assert key_source("openai", None, {}) == "OPENAI_API_KEY missing"


def test_a_wired_cortex_reads_its_key_from_the_environment_only(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    add_profile(environ, Vault(environ), profile("a", "anthropic"), KEY, make_active=True)
    with pytest.raises(KeyMissing, match="ANTHROPIC_API_KEY"):
        build_substrate(wired(), environ)


def test_a_local_model_needs_no_key_and_sends_no_authorization(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    add_profile(environ, Vault(environ), profile("local", "compatible"), None, make_active=True)
    substrate = build_substrate(default_config(), environ)
    assert substrate._api_key == ""  # type: ignore[attr-defined]
    assert "Authorization" not in _openai_headers("")
    assert _openai_headers(KEY)["Authorization"] == f"Bearer {KEY}"


def test_missing_saved_keys_are_named(tmp_path: Path, fake_keyring_and_home: FakeKeyring) -> None:
    environ = env_for(tmp_path)
    add_profile(environ, Vault(environ), profile(), KEY, make_active=True)
    fake_keyring_and_home.store.clear()
    with pytest.raises(KeyMissing, match="is gone") as caught:
        build_substrate(default_config(), environ)
    assert KEY not in str(caught.value)
    add_profile(environ, Vault(environ), profile(), None, make_active=True)
    with pytest.raises(KeyMissing, match="keeps no key"):
        build_substrate(default_config(), environ)
    with pytest.raises(InputRejected, match="angularai setup"):
        build_substrate(default_config(), env_for(tmp_path / "none"))


def test_the_saved_key_in_use_is_redacted_like_an_environment_secret(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    assert secret_values(environ) == ()
    add_profile(environ, Vault(environ), profile(), KEY, make_active=True)
    assert KEY in secret_values(environ)
    assert secret_values({**environ, "OPENAI_API_KEY": KEY}).count(KEY) == 1
    (tmp_path / "home" / MODELS_FILE).write_text("broken")
    assert secret_values(environ) == ()


def test_anything_configured(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    assert not anything_configured(environ)
    assert anything_configured({**environ, "ANTHROPIC_API_KEY": "x"})
    assert anything_configured({**environ, "ANGULARAI_MODEL": "x"})
    add_profile(environ, Vault(environ), profile("local", "compatible"), None, make_active=True)
    assert anything_configured(environ)

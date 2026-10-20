"""The setup and model reducers: every step, every key, and the key never in a frame."""

from collections.abc import Iterable
from dataclasses import replace

import pytest

from angularai.onboard import CHOICES, OLLAMA_URL
from angularai.profiles import mask
from angularai.setup_state import (
    BULLET,
    MANAGE_KEYS,
    SETUP_KEYS,
    SPINNER,
    Add,
    Event,
    Fetched,
    FetchModels,
    Finish,
    Key,
    Leave,
    ManageState,
    Paste,
    Probe,
    Probed,
    Refreshed,
    Remove,
    Row,
    Save,
    Saved,
    SaveFailed,
    SetupState,
    Step,
    Use,
    bullets,
    manage_start,
    manage_step,
    plain,
    render_manage,
    render_setup,
    setup_step,
    start,
    tick,
)
from tests.setupfakes import KEY

OPENAI, ANTHROPIC, OLLAMA, COMPATIBLE = range(4)


def key(name: str, char: str = "") -> Key:
    return Key(name, char)


def char(c: str) -> Key:
    return Key(c, c)


def drive(state: SetupState, events: Iterable[Event]) -> tuple[SetupState, list[object]]:
    effects: list[object] = []
    for event in events:
        state, effect = setup_step(state, event)
        if effect is not None:
            effects.append(effect)
    return state, effects


def typed(text: str) -> list[Event]:
    return [char(c) for c in text]


def frame(state: SetupState, first_use: bool = False) -> str:
    return plain(render_setup(state, first_use=first_use))


# --- provider -----------------------------------------------------------------


def test_provider_moves_with_arrows_and_j_k_and_stays_in_bounds() -> None:
    state = start()
    state, _ = drive(state, [key("up"), char("k")])
    assert state.cursor == 0
    state, _ = drive(state, [key("down"), char("j"), key("down"), key("down"), char("j")])
    assert state.cursor == len(CHOICES) - 1
    state, _ = drive(state, [key("home")])
    assert state.cursor == 0
    state, _ = drive(state, [key("end")])
    assert state.cursor == len(CHOICES) - 1
    state, effects = drive(state, [key("x", "x"), Fetched(("a",)), Probed(True)])
    assert (state.step, effects) == ("provider", [])


@pytest.mark.parametrize("name", ["escape", "q", "ctrl+c"])
def test_provider_leaves_on_escape_q_and_ctrl_c(name: str) -> None:
    state, effect = setup_step(start(), Key(name, "q" if name == "q" else ""))
    assert state.step == "cancelled"
    assert effect == Finish(saved=False)
    assert "Nothing was saved." in frame(state)


def test_a_digit_picks_a_provider() -> None:
    state, effect = setup_step(start(), char("2"))
    assert (state.step, state.choice.flag, effect) == ("key", "anthropic", None)
    state, _ = setup_step(start(), char("9"))
    assert state.step == "provider"
    state, _ = setup_step(start(), Paste("text"))
    assert state.step == "provider"


def test_first_use_says_nothing_is_set_up_and_the_provider_list_is_drawn() -> None:
    shown = frame(start(), first_use=True)
    assert "No model is set up yet." in shown
    assert "step 1 of 4" in shown
    for word in ("OpenAI", "Anthropic", "Ollama", "OpenAI-compatible", "OPENAI_API_KEY"):
        assert word in shown
    assert "No model is set up yet." not in frame(start())
    compatible = SetupState(cursor=COMPATIBLE)
    assert "step 1 of 5" in frame(compatible)
    assert "step 1 of 3" in frame(SetupState(cursor=OLLAMA))


# --- base url -----------------------------------------------------------------


def test_compatible_asks_a_base_url_and_checks_it() -> None:
    state, _ = drive(start(), [key("end"), key("enter")])
    assert state.step == "base_url"
    assert "http://127.0.0.1:1234/v1" in frame(state)
    state, _ = drive(state, [*typed("ftp://x"), key("enter")])
    assert state.step == "base_url" and state.error
    assert state.error in frame(state)
    state, _ = drive(state, [key("ctrl+u"), *typed("http://h:1/v1x"), key("backspace")])
    assert state.base_url == "http://h:1/v1" and state.error == ""
    state, _ = drive(state, [key("tab"), key("enter")])
    assert state.step == "key"
    assert "base url  http://h:1/v1" in frame(state)
    state, _ = drive(state, [key("escape")])
    assert state.step == "base_url"
    state, _ = drive(state, [Fetched(()), key("escape")])
    assert state.step == "provider"


def test_a_pasted_url_loses_its_newlines() -> None:
    state, _ = drive(SetupState(cursor=COMPATIBLE, step="base_url"), [Paste("http://h/v1\r\n")])
    assert state.base_url == "http://h/v1"


# --- key ----------------------------------------------------------------------


def test_the_key_shows_as_bullets_then_as_its_mask_and_never_in_clear() -> None:
    state, _ = drive(start(), [key("enter")])
    assert state.step == "key"
    assert "Paste works" in frame(state)
    frames = []
    for c in KEY:
        state, _ = setup_step(state, char(c))
        frames.append(frame(state))
    assert state.key == KEY
    assert BULLET * len(KEY) in frames[-1]
    state, effects = drive(state, [key("enter")])
    assert state.step == "model"
    assert effects == []
    frames.append(frame(state))
    assert mask(KEY) in frames[-1]
    for shown in frames:
        assert KEY not in shown
        assert KEY[3:12] not in shown
    assert KEY not in repr(state)


def test_bullets_stop_at_thirty_two() -> None:
    assert bullets("x" * 100) == BULLET * 32
    assert bullets("") == ""


def test_a_pasted_key_is_one_line_and_hidden() -> None:
    state = SetupState(step="key")
    state, _ = setup_step(state, Paste(KEY + "\n"))
    assert state.key == KEY
    assert KEY not in frame(state)
    assert KEY not in repr(Paste(KEY))


def test_openai_needs_a_key_unless_the_environment_has_one() -> None:
    state, _ = drive(start(), [key("enter"), key("enter")])
    assert state.step == "key"
    assert state.error == "OpenAI needs a key."
    with_env = start(frozenset({"openai"}))
    state, _ = drive(with_env, [key("enter")])
    assert "OPENAI_API_KEY and saves no key" in frame(state)
    state, _ = drive(state, [key("enter")])
    assert state.step == "model"
    assert "key       OPENAI_API_KEY" in frame(state)


def test_a_bad_key_is_refused_without_echoing_it() -> None:
    state, _ = drive(SetupState(step="key"), [*typed("bad key"), key("enter")])
    assert state.step == "key"
    assert state.error
    assert "bad key" not in state.error


def test_escape_on_the_key_goes_back_and_other_keys_do_nothing() -> None:
    state, _ = drive(SetupState(step="key"), [key("f5"), Fetched(())])
    assert state.step == "key"
    state, _ = drive(state, [key("escape")])
    assert state.step == "provider"


def test_compatible_with_no_key_lists_the_models() -> None:
    state = SetupState(step="key", cursor=COMPATIBLE, base_url="http://h/v1")
    assert "server without a key" in frame(state)
    state, effect = setup_step(state, key("enter"))
    assert state.step == "model" and state.fetching
    assert effect == FetchModels(CHOICES[COMPATIBLE], "http://h/v1", "")
    assert "listing models at http://h/v1" in frame(state)
    state, _ = setup_step(state, Fetched(("b-model", "a-model")))
    assert (state.models, state.fetching, state.listed) == (("b-model", "a-model"), False, True)
    state, _ = setup_step(state, Fetched(("late",)))
    assert state.models == ("b-model", "a-model")


def test_a_compatible_key_rides_along_to_the_listing_but_not_the_repr() -> None:
    state = SetupState(step="key", cursor=COMPATIBLE, base_url="http://h/v1", key=KEY)
    _, effect = setup_step(state, key("enter"))
    assert isinstance(effect, FetchModels) and effect.key == KEY
    assert KEY not in repr(effect)


def test_no_listing_falls_back_to_free_text() -> None:
    state = SetupState(step="key", cursor=COMPATIBLE, base_url="http://h/v1")
    state, _ = drive(state, [key("enter"), Fetched(())])
    assert state.on_custom
    assert "No model list at that endpoint" in frame(state)
    state, effects = drive(state, [*typed("my-jk-q"), key("enter")])
    assert state.model == "my-jk-q"
    assert effects == [Probe(CHOICES[COMPATIBLE], "my-jk-q", "http://h/v1", "")]


def test_ollama_skips_url_and_key() -> None:
    state, effects = drive(start(), [char("3")])
    assert state.step == "model"
    assert state.base_url == OLLAMA_URL
    assert effects == [FetchModels(CHOICES[OLLAMA], OLLAMA_URL, "")]
    assert "llama3.2" in frame(state)
    assert "key  " not in frame(state)
    state, _ = drive(state, [key("escape")])
    assert state.step == "provider"


# --- model --------------------------------------------------------------------


def test_model_picks_a_suggestion_or_types_one() -> None:
    state = SetupState(step="model", cursor=OPENAI, key=KEY, models=("m1", "m2"), listed=True)
    state, _ = drive(state, [char("j")])
    assert state.model_cursor == 1
    state, _ = drive(state, [key("down")])
    assert state.on_custom
    assert "other: type a model id" not in frame(state)
    state, _ = drive(state, [*typed("jkq"), Paste("x")])
    assert state.custom == "jkqx"
    state, _ = drive(state, [key("up")])
    assert state.model_cursor == 1
    assert "other: type a model id" in frame(state)
    state, effects = drive(state, [key("enter")])
    assert state.step == "test" and state.model == "m2"
    assert effects == [Probe(CHOICES[OPENAI], "m2", None, KEY)]


def test_model_refuses_an_empty_id_and_q_quits_off_the_text_row() -> None:
    state = SetupState(step="model", models=(), listed=True)
    state, _ = drive(state, [key("enter")])
    assert state.step == "model" and state.error
    state = SetupState(step="model", models=("m1",), listed=True)
    state, effects = drive(state, [key("f2")])
    assert effects == []
    state, effects = drive(state, [char("q")])
    assert state.step == "cancelled" and effects == [Finish(saved=False)]


def test_escape_on_the_model_goes_back_to_the_key() -> None:
    state, _ = drive(SetupState(step="model", models=("m",)), [key("escape")])
    assert state.step == "key"


def test_listed_models_are_kept_when_coming_back() -> None:
    state = SetupState(
        step="key", cursor=COMPATIBLE, base_url="http://h/v1", models=("x",), listed=True
    )
    state, effect = setup_step(state, key("enter"))
    assert effect is None and state.models == ("x",)


# --- test ---------------------------------------------------------------------


def at_test(**values: object) -> SetupState:
    base = SetupState(step="test", cursor=OPENAI, key=KEY, model="m", test="running")
    return replace(base, **values)  # type: ignore[arg-type]


def test_the_test_spins_then_says_ok() -> None:
    state = at_test()
    assert "testing api.openai.com" in frame(state)
    assert SPINNER[0] in frame(state)
    assert SPINNER[1] in frame(tick(state))
    ignored, effect = setup_step(state, key("enter"))
    assert ignored == state and effect is None
    state, _ = setup_step(state, Probed(True, seconds=0.25))
    assert state.test == "ok"
    shown = frame(state)
    assert "✓ Connected in 0.2 s." in shown or "✓ Connected in 0.3 s." in shown
    for option in ("Save", "Edit", "Cancel"):
        assert option in shown
    assert "Retry" not in shown
    assert KEY not in shown and mask(KEY) in shown
    late, _ = setup_step(state, Probed(False, "late"))
    assert late.test == "ok"


def test_anthropic_tests_its_own_host() -> None:
    assert "testing api.anthropic.com" in frame(at_test(cursor=ANTHROPIC))
    assert "testing http://h/v1" in frame(at_test(cursor=COMPATIBLE, base_url="http://h/v1"))


def test_save_after_ok_and_the_saved_frame() -> None:
    state, effect = setup_step(at_test(test="ok"), key("enter"))
    assert state.test == "saving"
    assert effect == Save(CHOICES[OPENAI], "m", None, KEY)
    assert KEY not in repr(effect)
    assert "saving" in frame(state)
    state, effect = setup_step(state, key("enter"))
    assert effect is None
    state, effect = setup_step(state, Saved("openai", "Key sk-…a1b2 in keys.json."))
    assert state.step == "saved" and effect == Finish(saved=True)
    assert "Saved openai." in frame(state)
    assert SETUP_KEYS["saved"] == ""


def test_a_failed_save_comes_back_as_a_failure() -> None:
    state, _ = setup_step(at_test(test="saving"), SaveFailed("disk full"))
    assert (state.test, state.message) == ("failed", "disk full")
    assert "✗ disk full" in frame(state)


def test_a_failed_test_offers_retry_edit_save_anyway_cancel() -> None:
    state, _ = setup_step(at_test(), Probed(False, "The key was refused."))
    shown = frame(state)
    assert "✗ The key was refused." in shown
    for option in ("Retry", "Edit", "Save anyway", "Cancel"):
        assert option in shown
    retried, effect = setup_step(state, char("r"))
    assert retried.test == "running" and isinstance(effect, Probe)
    saved, effect = setup_step(state, char("s"))
    assert saved.test == "saving" and isinstance(effect, Save)
    edited, effect = setup_step(state, char("e"))
    assert edited.step == "provider" and edited.key == KEY and effect is None
    for quit_key in ("c", "q"):
        gone, effect = setup_step(state, char(quit_key))
        assert gone.step == "cancelled" and effect == Finish(saved=False)
    back, _ = setup_step(state, key("escape"))
    assert back.step == "model"
    moved, _ = drive(state, [key("down"), key("down"), key("down"), key("down")])
    assert moved.option == 3
    gone, effect = setup_step(moved, key("enter"))
    assert effect == Finish(saved=False)
    same, effect = setup_step(state, char("z"))
    assert same == state and effect is None
    assert "The test failed." in frame(at_test(test="failed", message=""))


def test_save_anyway_with_no_key_saves_none() -> None:
    _, effect = setup_step(at_test(test="ok", key=""), char("s"))
    assert effect == Save(CHOICES[OPENAI], "m", None, None)


def test_retry_is_not_offered_after_ok() -> None:
    state = at_test(test="ok")
    same, effect = setup_step(state, char("r"))
    assert same == state and effect is None


def test_ctrl_c_cancels_from_every_step() -> None:
    steps: tuple[Step, ...] = ("provider", "base_url", "key", "model", "test")
    for step in steps:
        state, effect = setup_step(SetupState(step=step), key("ctrl+c"))
        assert state.step == "cancelled" and effect == Finish(saved=False)
    done = SetupState(step="saved")
    assert setup_step(done, key("enter")) == (done, None)


def test_every_step_has_its_keys_line() -> None:
    for step in ("provider", "base_url", "key", "model", "test"):
        assert SETUP_KEYS[step]


# --- manage -------------------------------------------------------------------


def rows() -> tuple[Row, ...]:
    return (
        Row("openai", "openai", "gpt-5-mini", mask(KEY) + " in keyring", "", False),
        Row("local", "compatible", "llama3.2", "no key", "http://h/v1", True),
    )


def mstep(state: ManageState, *events: Key | Refreshed) -> tuple[ManageState, list[object]]:
    effects: list[object] = []
    for event in events:
        state, effect = manage_step(state, event)
        if effect is not None:
            effects.append(effect)
    return state, effects


def test_manage_starts_on_the_active_row_and_marks_it() -> None:
    state = manage_start(rows(), "ANGULARAI_MODEL picks x in this shell.")
    assert state.cursor == 1
    shown = plain(render_manage(state))
    assert "● local" in shown
    assert mask(KEY) in shown and KEY not in shown
    assert "http://h/v1" in shown
    assert "ANGULARAI_MODEL picks x" in shown
    assert MANAGE_KEYS.startswith("↑↓ move")


def test_manage_moves_uses_adds_and_leaves() -> None:
    state = manage_start(rows())
    state, _ = mstep(state, char("k"))
    assert state.cursor == 0
    state, _ = mstep(state, key("up"), key("down"), char("j"), key("home"))
    assert state.cursor == 0
    for use in (key("enter"), char("u"), Key("space", " ")):
        _, effects = mstep(state, use)
        assert effects == [Use("openai")]
    _, effects = mstep(state, char("a"))
    assert effects == [Add()]
    for leave in (key("escape"), char("q"), key("ctrl+c")):
        _, effects = mstep(state, leave)
        assert effects == [Leave()]
    same, effects = mstep(state, key("f1"))
    assert same == state and effects == []


def test_manage_removes_only_after_y() -> None:
    state = manage_start(rows())
    for ask in (char("d"), char("x"), key("delete"), key("backspace")):
        asking, _ = mstep(state, ask)
        assert asking.confirm == "local"
        assert "Remove local and its key? y / n" in plain(render_manage(asking))
    asking, _ = mstep(state, char("d"))
    kept, effects = mstep(asking, char("n"))
    assert effects == [] and kept.confirm is None
    assert "Nothing was removed." in plain(render_manage(kept))
    _, effects = mstep(asking, char("Y"))
    assert effects == [Remove("local")]
    _, effects = mstep(asking, key("ctrl+c"))
    assert effects == [Leave()]


def test_manage_refresh_keeps_the_cursor_in_range_and_says_why() -> None:
    state = manage_start(rows())
    state, _ = mstep(state, Refreshed(rows()[:1], "Removed local."))
    assert state.cursor == 0
    assert "Removed local." in plain(render_manage(state))
    state, _ = mstep(state, Refreshed((), ""))
    assert "No saved models. Press a to add one." in plain(render_manage(state))
    same, effects = mstep(state, key("enter"))
    assert effects == [] and same == state
    _, effects = mstep(state, char("a"))
    assert effects == [Add()]
    assert manage_start(()).cursor == 0

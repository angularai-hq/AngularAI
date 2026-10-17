from pathlib import Path

import pytest

from angularai.budget import estimate_tokens
from angularai.errors import SubstrateUnavailable
from angularai.models import Integration, Workspace
from angularai.pilot import pass_turn
from angularai.store import Store, init_cortex, open_store
from tests.scripted import Scripted, noted, said


def cortex(tmp_path: Path) -> Store:
    init_cortex(tmp_path)
    return open_store(tmp_path)


def test_a_long_episode_shows_the_matching_span(tmp_path: Path) -> None:
    store = cortex(tmp_path)
    head = "The warehouse counted cedar planks at dawn. " * 30
    brief = (
        head
        + "Elena Rostova was replaced by Marcus Vance, badge MV-882-K, as the secondary contact. "
        + "The catering order is 14 boxes of vegan pastries."
    )
    pass_turn(store, Scripted(said("Noted.")), brief)
    question = Scripted(said("Marcus Vance, MV-882-K."))
    pass_turn(store, question, "Who is the secondary contact and what is their badge?")
    shown = question.seen[-1].user
    assert "Marcus Vance" in shown
    assert "MV-882-K" in shown
    assert "cedar planks" not in shown


def test_early_fact_returns_inside_a_bounded_workspace(tmp_path: Path) -> None:
    store = cortex(tmp_path)
    fact = Scripted(noted("The notebook is blue.", "place"))
    pass_turn(store, fact, "I have a blue notebook.")

    fillers = [f"note-{index:03d} " + ("cedar plank " * 20) for index in range(40)]
    filler_model = Scripted(said("Ok."))
    for filler in fillers:
        pass_turn(store, filler_model, filler)

    question = Scripted(said("The notebook is blue."))
    result = pass_turn(store, question, "What color is the notebook?")
    workspace = question.seen[-1]

    assert result.reply == "The notebook is blue."
    assert workspace.estimated_tokens <= workspace.budget
    assert workspace.estimated_tokens == estimate_tokens(workspace.system) + estimate_tokens(
        workspace.user
    )
    assert "The notebook is blue." in workspace.user
    assert "note-000" not in workspace.user
    assert estimate_tokens(fillers[0]) + workspace.estimated_tokens <= workspace.budget
    assert workspace.episodes_held > 0

    life = estimate_tokens("\n".join(episode.user for episode in store.view().episodes))
    assert life > store.config().workspace_budget
    assert "note-000" in (tmp_path / "passes.jsonl").read_text(encoding="utf-8")
    for seen in (*fact.seen, *filler_model.seen, *question.seen):
        assert seen.estimated_tokens <= seen.budget


def test_a_trace_the_model_did_not_see_cannot_be_superseded(tmp_path: Path) -> None:
    store = cortex(tmp_path)
    pass_turn(store, Scripted(noted("The notebook is blue.", "place")), "I have a blue notebook.")
    pass_turn(
        store,
        Scripted(noted("Warehouse counted cedar planks.", "warehouse")),
        "The warehouse counted cedar planks.",
    )
    traces = {trace.text: trace for trace in store.view().current_traces()}
    hidden = traces["Warehouse counted cedar planks."]
    visible = traces["The notebook is blue."]

    class Asking:
        def integrate(self, workspace: Workspace) -> Integration:
            assert hidden.id not in workspace.trace_ids
            assert visible.id in workspace.trace_ids
            return Integration("The notebook is blue.", (), (visible.id, hidden.id))

    pass_turn(store, Asking(), "What color is the notebook?")
    view = store.view()
    by_id = {trace.id: trace for trace in view.traces}
    assert by_id[visible.id].state == "superseded"
    assert by_id[hidden.id].state == "current"
    log = (tmp_path / "passes.jsonl").read_text(encoding="utf-8")
    assert "The notebook is blue." in log


def test_a_failed_call_writes_no_trace(tmp_path: Path) -> None:
    store = cortex(tmp_path)
    pass_turn(store, Scripted(noted("The notebook is blue.", "place")), "I have a blue notebook.")
    before = store.view().current_traces()

    class Down:
        def integrate(self, workspace: Workspace) -> Integration:
            raise SubstrateUnavailable("The substrate is unreachable.")

    with pytest.raises(SubstrateUnavailable, match="unreachable"):
        pass_turn(store, Down(), "Tell me about the river.")
    assert store.view().current_traces() == before
    failed = store.view().episodes[-1]
    assert failed.outcome == "error"
    assert failed.reply is None
    assert failed.user == "Tell me about the river."

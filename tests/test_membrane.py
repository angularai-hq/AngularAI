from pathlib import Path

import pytest

from clasp.budget import estimate_tokens
from clasp.errors import SubstrateUnavailable
from clasp.models import Integration, Workspace
from clasp.pilot import pass_turn
from clasp.store import Store, init_cortex, open_store
from tests.scripted import Scripted, noted, said


def cortex(tmp_path: Path) -> Store:
    init_cortex(tmp_path)
    return open_store(tmp_path)


def test_early_fact_returns_inside_a_bounded_workspace(tmp_path: Path) -> None:
    store = cortex(tmp_path)
    fact = Scripted(noted("Lives in Nitra.", "place"))
    pass_turn(store, fact, "I live in Nitra.")

    fillers = [f"note-{index:03d} " + ("cedar plank " * 20) for index in range(40)]
    filler_model = Scripted(said("Ok."))
    for filler in fillers:
        pass_turn(store, filler_model, filler)

    question = Scripted(said("You live in Nitra."))
    result = pass_turn(store, question, "Where do I live?")
    workspace = question.seen[-1]

    assert result.reply == "You live in Nitra."
    assert workspace.estimated_tokens <= workspace.budget
    assert workspace.estimated_tokens == estimate_tokens(workspace.system) + estimate_tokens(
        workspace.user
    )
    assert "Lives in Nitra." in workspace.user
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
    pass_turn(store, Scripted(noted("Lives in Nitra.", "place")), "I live in Nitra.")
    pass_turn(
        store,
        Scripted(noted("Warehouse counted cedar planks.", "warehouse")),
        "The warehouse counted cedar planks.",
    )
    traces = {trace.text: trace for trace in store.view().current_traces()}
    hidden = traces["Warehouse counted cedar planks."]
    visible = traces["Lives in Nitra."]

    class Asking:
        def integrate(self, workspace: Workspace) -> Integration:
            assert hidden.id not in workspace.trace_ids
            assert visible.id in workspace.trace_ids
            return Integration("You live in Nitra.", (), (visible.id, hidden.id))

    pass_turn(store, Asking(), "Where do I live?")
    view = store.view()
    by_id = {trace.id: trace for trace in view.traces}
    assert by_id[visible.id].state == "superseded"
    assert by_id[hidden.id].state == "current"
    log = (tmp_path / "passes.jsonl").read_text(encoding="utf-8")
    assert "Lives in Nitra." in log


def test_a_failed_call_writes_no_trace(tmp_path: Path) -> None:
    store = cortex(tmp_path)
    pass_turn(store, Scripted(noted("Lives in Nitra.", "place")), "I live in Nitra.")
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

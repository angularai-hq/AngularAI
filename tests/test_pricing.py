import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from angularai import run as run_loop
from angularai.errors import InputRejected, ServerError
from angularai.models import Integration, Workspace
from angularai.pricing import PRICES, Price, Usage, parse_price, price_for, usage_of
from tests.scripted import Scripted
from tests.test_run import TASK, act, begin, compatible, finished, message, native_workspace

PRICE = Price(input=1.0, output=10.0, cached=0.1)


class Billed(Scripted):
    """A scripted substrate that is billed a fixed usage for every call."""

    def __init__(self, *script: Integration, per_call: Usage = Usage(1000, 0, 100)) -> None:
        super().__init__(script=script, reply=act("list", path="."))
        self.per_call = per_call
        self.total = Usage()

    def integrate(self, workspace: Workspace) -> Integration:
        self.total = self.total + self.per_call
        return super().integrate(workspace)

    def billed(self) -> Usage:
        return self.total


def test_usage_reads_openai_and_anthropic_blocks() -> None:
    openai = {
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 80,
            "prompt_tokens_details": {"cached_tokens": 1024},
        }
    }
    assert usage_of(openai) == Usage(1200, 1024, 80)
    anthropic = {
        "usage": {
            "input_tokens": 50,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 10,
            "output_tokens": 7,
        }
    }
    assert usage_of(anthropic) == Usage(960, 900, 7)
    assert usage_of({}) == Usage()
    assert usage_of({"usage": {"prompt_tokens": -3, "completion_tokens": True}}) == Usage()


def test_price_counts_cached_input_at_the_cached_rate() -> None:
    assert PRICE.cost(Usage(1_000_000, 0, 0)) == pytest.approx(1.0)
    assert PRICE.cost(Usage(1_000_000, 1_000_000, 0)) == pytest.approx(0.1)
    assert PRICE.cost(Usage(0, 0, 100_000)) == pytest.approx(1.0)
    mini = price_for("gpt-5.4-mini")
    assert mini == PRICES["gpt-5.4-mini"] == Price(0.75, 4.50, 0.075)
    assert price_for("aa-qwen3-4b") is None
    assert parse_price(" 1, 10 ,0.1") == PRICE
    with pytest.raises(ValueError):
        parse_price("1,2")
    with pytest.raises(ValueError):
        parse_price("1,-2,0")


def test_the_compatible_substrate_adds_up_usage() -> None:
    sent: list[Mapping[str, object]] = []
    first = message("", [{"function": {"name": "list", "arguments": "{}"}}])
    first["usage"] = {"prompt_tokens": 100, "completion_tokens": 5}
    second = message("", [{"function": {"name": "list", "arguments": "{}"}}])
    second["usage"] = {
        "prompt_tokens": 200,
        "completion_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 64},
    }
    substrate = compatible([(200, first), (200, second)], sent)
    assert substrate.billed() == Usage()
    substrate.integrate(native_workspace())
    substrate.integrate(native_workspace())
    assert substrate.billed() == Usage(300, 64, 12)


def test_each_step_is_priced_and_logged(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, price=PRICE, max_cost=1.0)
    scripted = Billed(act("read", path="notes.txt"), finished())
    done = run_loop.advance(runs, state, store, scripted)
    # 1000 input and 100 output tokens: 0.001 + 0.001 dollars a step.
    assert done.status == "done" and done.cost == pytest.approx(0.004)
    assert done.billed == (2000, 0, 200)
    lines = [
        json.loads(line) for line in (runs / state.id / "steps.jsonl").read_text().splitlines()
    ]
    assert [line["cost"] for line in lines] == [0.002, 0.002]
    assert lines[-1]["total_cost"] == 0.004
    assert run_loop.load(runs, state.id) == done


def test_the_guard_stops_before_a_step_that_could_pass_the_limit(tmp_path: Path) -> None:
    # The worst case at budget 1000 is 3500 input and 2048 output tokens: $0.02398.
    runs, state, store = begin(tmp_path, budget=1000, price=PRICE, max_cost=0.03)
    assert run_loop.worst_step(state) == pytest.approx(0.02398)
    lines: list[str] = []
    stopped = run_loop.advance(runs, state, store, Billed(), progress=lines.append)
    # Each step costs $0.002; the next is estimated at 1.25 times the dearest.
    assert stopped.status == "cost" and stopped.step == 14
    assert stopped.cost == pytest.approx(0.028)
    assert stopped.cost + run_loop.next_step(stopped) > 0.03
    assert stopped.cost <= 0.03
    assert lines[-1] == "Stopped before step 15: $0.0280 spent."
    with pytest.raises(InputRejected, match="nothing to resume"):
        run_loop.advance(runs, stopped, store, Billed())
    raised = run_loop.raise_cost(stopped, 0.05)
    assert raised.status == "running" and raised.max_cost == 0.05
    with pytest.raises(InputRejected):
        run_loop.raise_cost(stopped, 0.01)


def test_the_first_step_must_fit_the_worst_case(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, budget=1000, price=PRICE, max_cost=0.02)
    scripted = Billed()
    stopped = run_loop.advance(runs, state, store, scripted)
    assert (stopped.status, stopped.step, stopped.cost) == ("cost", 0, 0.0)
    assert scripted.seen == []


def test_a_step_without_usage_is_charged_the_worst_case(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, budget=1000, price=PRICE, max_cost=1.0)
    done = run_loop.advance(runs, state, store, Billed(finished(), per_call=Usage()))
    assert done.cost == pytest.approx(run_loop.worst_step(state))


def test_a_failed_step_still_pays_for_what_it_used(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, price=PRICE, max_cost=1.0)

    class Down(Billed):
        def integrate(self, workspace: Workspace) -> Integration:
            self.total = self.total + Usage(500, 0, 0)
            raise ServerError("The model server failed.", 503)

    with pytest.raises(ServerError):
        run_loop.advance(runs, state, store, Down())
    saved = run_loop.load(runs, state.id)
    assert saved.step == 0 and saved.cost == pytest.approx(0.0005)


def test_a_cost_limit_needs_a_price_and_a_meter(tmp_path: Path) -> None:
    with pytest.raises(InputRejected, match="needs a price"):
        begin(tmp_path, max_cost=1.0)
    runs, state, store = begin(tmp_path / "b", price=PRICE, max_cost=1.0)
    with pytest.raises(InputRejected, match="no usage"):
        run_loop.advance(runs, state, store, Scripted(finished()))
    with pytest.raises(InputRejected, match="cost limit"):
        begin(tmp_path / "c", price=PRICE, max_cost=0)
    free = begin(tmp_path / "d")[1]
    with pytest.raises(InputRejected, match="no price"):
        run_loop.raise_cost(free, 1.0)
    priced = run_loop.advance(*begin(tmp_path / "e", price=PRICE)[::1], Billed(finished()))
    assert priced.cost == pytest.approx(0.002) and priced.max_cost is None


def test_the_cli_prices_a_known_model_and_takes_a_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from angularai.cli import main

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("ANGULARAI_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANGULARAI_INDEX", raising=False)
    scripted = Billed()
    monkeypatch.setattr("angularai.cli.build_substrate", lambda config, environ: scripted)
    cortex = ["--cortex", str(tmp_path / "cortex")]
    assert main([*cortex, "init"]) == 0
    base = [*cortex, "run", "--root", str(work), "--finish-check", "off"]
    # The cortex has no paid model, so a limit needs --price.
    assert main([*base, "--max-cost", "1", TASK]) == 1
    assert "needs a price" in capsys.readouterr().err
    assert main([*base, "--price", "1,x,0", "--max-cost", "1", TASK]) == 1
    capsys.readouterr()
    assert main([*base, "--price", "1,10,0.1", "--max-cost", "0.01", TASK]) == 0
    err = capsys.readouterr().err
    run_id = err.split("run   ")[1].split()[0]
    assert "The cost limit stops the run." in err and "cost  $0.0000" in err
    assert main(["run", "--resume", run_id, "--max-cost", "0.03", "--steps", "2"]) == 0
    err = capsys.readouterr().err
    assert "Paused at step 2." in err and "cost  $0.0040" in err


def test_a_step_that_fails_before_any_call_costs_nothing(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, price=PRICE, max_cost=1.0)

    class Refused(Billed):
        def integrate(self, workspace: Workspace) -> Integration:
            raise ServerError("The model server failed.", 503)

    with pytest.raises(ServerError):
        run_loop.advance(runs, state, store, Refused())
    assert run_loop.load(runs, state.id).cost == 0.0

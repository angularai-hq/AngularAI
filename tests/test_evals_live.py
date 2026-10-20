"""Live evals. Opt in: ANGULARAI_EVAL_LIVE=1 and a local server (Ollama by default).

ANGULARAI_EVAL_MODEL names the model; ANGULARAI_EVAL_BASE_URL the server.
"""

import os
from pathlib import Path

import pytest

from angularai.evals.client import Chat
from angularai.evals.runner import answer
from angularai.provider import Policy
from angularai.substrate import substrate_for
from tests.evalfakes import question

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("ANGULARAI_EVAL_LIVE") != "1", reason="set ANGULARAI_EVAL_LIVE=1"
    ),
]
BASE = os.environ.get("ANGULARAI_EVAL_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.environ.get("ANGULARAI_EVAL_MODEL", "aa-qwen2.5-3b")


def test_live_full_and_v1_answer(tmp_path: Path) -> None:
    chat = Chat(model=MODEL, base_url=BASE)
    full = answer("full", question(), chat, budget=2000)
    assert full["error"] is None and "blue" in str(full["prediction"]).lower()
    policy = Policy(read_timeout=600.0, deadline=600.0, attempts=1)
    substrate = substrate_for("compatible", MODEL, BASE, "", policy, json_mode=True)
    v1 = answer("v1", question(), chat, budget=2000, substrate=substrate)
    assert v1["error"] is None and v1["prediction"]

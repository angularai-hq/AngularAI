"""Score one smoke run: files done and correct, SUMMARY.md written and correct, steps, end."""

import json
import re
import sys
from pathlib import Path

WORDS = {
    3: ("3", "three", "triple"),
    4: ("4", "four", "quadruple"),
    5: ("5", "five"),
    7: ("7", "seven"),
}
work, run = Path(sys.argv[1]), Path(sys.argv[2])
names = sorted(p.stem for p in (work / "src").glob("*.py"))


def right(text: str, k: int) -> bool:
    low = text.lower()
    return any(re.search(rf"\b{w}", low) for w in WORDS[k]) and "squar" not in low


done = correct = 0
for name in names:
    doc = (work / "src" / f"{name}.py").read_text()
    m = re.search(r'"""(.*?)"""', doc, re.S)
    body = m.group(1) if m else ""
    ok_shape = doc.startswith(f"def {name}(x):") and f"return x * {len(name)}" in doc
    if body and "TODO" not in body and ok_shape:
        done += 1
        correct += right(body, len(name))
summary = work / "SUMMARY.md"
listed = summary_right = 0
if summary.is_file():
    for name in names:
        lines = [
            line for line in summary.read_text().splitlines() if re.search(rf"\b{name}\b", line)
        ]
        if lines:
            listed += 1
            summary_right += any(right(line, len(name)) for line in lines)
state = json.loads((run / "state.json").read_text())
steps = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
print(
    json.dumps(
        {
            "files_done": done,
            "files_correct": correct,
            "files": len(names),
            "summary": summary.is_file(),
            "summary_listed": listed,
            "summary_correct": summary_right,
            "steps": state["step"],
            "status": state["status"],
            "tokens_mean": round(sum(s["used"] for s in steps) / len(steps)),
            "seconds_mean": round(sum(s["seconds"] for s in steps) / len(steps), 1),
        }
    )
)

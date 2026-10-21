"""Score the long run: files, SUMMARY.md, tokens and quality by phase of the run."""

import ast
import json
import re
import statistics
import sys
from pathlib import Path

NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
OPS = {
    "*": ("multipl", "times", "*", "product", "scale"),
    "+": ("add", "plus", "+", "sum", "increase", "increment"),
}
work, run, spec = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())


# One word for the operation and its number: "doubles x" is x * 2.
WHOLE = {("*", 2): ("double", "twice"), ("*", 3): ("triple", "thrice"), ("*", 4): ("quadruple",)}


def right(text: str, op: str, k: int) -> bool:
    low = text.lower()
    if any(word in low for word in WHOLE.get((op, k), ())):
        return True
    number = re.search(rf"\b({k}|{NUMBERS[k]})\b", low) is not None
    other = "+" if op == "*" else "*"
    return number and any(w in low for w in OPS[op]) and not any(w in low for w in OPS[other][:2])


def docstring(source: str) -> str:
    """The function's docstring in any quoting; empty when there is none or it breaks."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return ast.get_docstring(node) or ""
    return ""


steps = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
state = json.loads((run / "state.json").read_text())
finished_at: dict[str, int] = {}
for s in steps:
    m = re.match(r"(?:EDITED|WROTE) src/(\w+)\.py:", s.get("change", ""))
    if m:
        finished_at[m.group(1)] = s["step"]
files = {}
for name, f in spec.items():
    doc = (work / "src" / f"{name}.py").read_text()
    body = docstring(doc)
    intact = doc.startswith(f"def {name}(x):") and f"return x {f['op']} {f['k']}" in doc
    done = bool(body) and "TODO" not in body and intact
    files[name] = {
        "done": done,
        "correct": done and right(body, f["op"], f["k"]),
        "step": finished_at.get(name),
    }
summary = work / "SUMMARY.md"
listed = correct_lines = 0
if summary.is_file():
    text = summary.read_text().splitlines()
    for name, f in spec.items():
        hits = [line for line in text if re.search(rf"\b{name}\b", line)]
        listed += bool(hits)
        correct_lines += any(right(line, f["op"], f["k"]) for line in hits)
used = [s["used"] for s in steps]
thirds = [steps[i * len(steps) // 3 : (i + 1) * len(steps) // 3] for i in range(3)]
done_files = sorted((f["step"], f["correct"]) for f in files.values() if f["done"] and f["step"])
halves = [done_files[: len(done_files) // 2], done_files[len(done_files) // 2 :]]
out = {
    "run": state["id"],
    "status": state["status"],
    "steps": state["step"],
    "max_steps": state["max_steps"],
    "budget": state["budget"],
    "contiguous": [s["step"] for s in steps] == list(range(1, len(steps) + 1)),
    "files": len(spec),
    "files_done": sum(f["done"] for f in files.values()),
    "files_correct": sum(f["correct"] for f in files.values()),
    "summary": summary.is_file(),
    "summary_listed": listed,
    "summary_correct": correct_lines,
    "tokens": {
        "mean": round(statistics.mean(used)),
        "median": statistics.median(used),
        "max": max(used),
        "over_budget": sum(s["used"] > s["budget"] for s in steps),
    },
    "tokens_by_third": [round(statistics.mean(s["used"] for s in t)) for t in thirds if t],
    "seconds": {
        "mean": round(statistics.mean(s["seconds"] for s in steps), 1),
        "total_min": round(sum(s["seconds"] for s in steps) / 60, 1),
    },
    "seconds_by_third": [round(statistics.mean(s["seconds"] for s in t), 1) for t in thirds if t],
    "correct_rate_first_half_done": round(sum(c for _, c in halves[0]) / max(1, len(halves[0])), 2),
    "correct_rate_second_half_done": round(
        sum(c for _, c in halves[1]) / max(1, len(halves[1])), 2
    ),
    "steps_per_file_by_half": [
        round((h[-1][0] - (h[0][0] if i == 0 else halves[0][-1][0])) / max(1, len(h)), 1)
        if h
        else None
        for i, h in enumerate(halves)
    ],
    "tools": {
        t: sum((s.get("tool") or "none") == t for s in steps)
        for t in sorted({s.get("tool") or "none" for s in steps})
    },
    "finish_checks": state.get("checks"),
}
print(json.dumps(out, indent=1))

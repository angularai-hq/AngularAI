"""The demo500 fixture: 130 one-function modules, each `x * k` or `x + k`, seeded."""

import json
import random
import sys
from pathlib import Path

ADJECTIVES = "amber brisk calm dusky eager fallow gentle hollow ivory jolly".split()
NOUNS = """badger cedar delta ember falcon garnet harbor iris juniper kestrel lagoon maple
nectar onyx pebble""".split()
root = Path(sys.argv[1])
rng = random.Random(500)
spec = {}
(root / "src").mkdir(parents=True)
# 130, not 150: the smoke took 3.3 steps a module, so 130 modules and the summary
# fill about 500 steps; 150 could not finish under --max-steps 500.
names = [f"{adjective}_{noun}" for adjective in ADJECTIVES for noun in NOUNS][:130]
for name in names:
    op = rng.choice(["*", "+"])
    k = rng.randint(2, 9)
    spec[name] = {"op": op, "k": k}
    (root / "src" / f"{name}.py").write_text(
        f'def {name}(x):\n    """TODO: describe."""\n    return x {op} {k}\n'
    )
(root / "README.txt").write_text("Each module in src/ holds one small function.\n")
Path(sys.argv[2]).write_text(json.dumps(spec, indent=1))
print(len(spec))

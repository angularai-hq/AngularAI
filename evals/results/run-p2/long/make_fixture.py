"""The long-run fixture: 48 one-function modules, each `x * k` or `x + k`, seeded."""

import json
import random
import sys
from pathlib import Path

WORDS = """amber basil cedar delta ember fable garnet harbor indigo juniper kestrel lagoon
maple nectar onyx pebble quartz raven saffron tundra umber velvet willow xenon yarrow
zephyr acorn birch comet dune ermine fjord glacier heron iris jasper koala lilac
meadow nimbus orchid poplar quill ripple sable thistle""".split()
root = Path(sys.argv[1])
rng = random.Random(13)
spec = {}
(root / "src").mkdir(parents=True)
for name in WORDS:
    op = rng.choice(["*", "+"])
    k = rng.randint(2, 9)
    spec[name] = {"op": op, "k": k}
    (root / "src" / f"{name}.py").write_text(
        f'def {name}(x):\n    """TODO: describe."""\n    return x {op} {k}\n'
    )
(root / "README.txt").write_text("Each module in src/ holds one small function.\n")
Path(sys.argv[2]).write_text(json.dumps(spec, indent=1))
print(len(spec))

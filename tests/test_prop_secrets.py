"""Properties of secret detection: every known key shape is caught, benign text is not."""

import random
import string
import uuid

from hypothesis import example, given
from hypothesis import strategies as st

from angularai.secrets import contains_secret, looks_secret, redact

ALNUM = string.ascii_letters + string.digits


def body(alphabet: str, low: int, high: int | None = None) -> st.SearchStrategy[str]:
    return st.text(alphabet=alphabet, min_size=low, max_size=high or low + 40)


def prefixed(prefixes: list[str], rest: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    return st.builds(lambda head, tail: head + tail, st.sampled_from(prefixes), rest)


jwt_part = body(ALNUM + "_-", 8)
pem_kind = st.sampled_from(["", "RSA ", "EC ", "OPENSSH ", "ENCRYPTED "])
pem_lines = st.lists(body(ALNUM + "+/=", 1, 64), min_size=1, max_size=5)

# One strategy per supported shape, with a random valid body.
KEYS: dict[str, st.SearchStrategy[str]] = {
    "openai/anthropic": prefixed(
        ["sk-", "sk-ant-", "sk-proj-", "sk-svcacct-", "sk-admin-"], body(ALNUM + "_-", 16)
    ),
    "stripe": prefixed(["sk_live_", "sk_test_", "rk_live_", "rk_test_"], body(ALNUM, 10)),
    "github pat": prefixed(["github_pat_"], body(ALNUM + "_", 22)),
    "github": prefixed(["ghp_", "gho_", "ghu_", "ghs_", "ghr_"], body(ALNUM, 20)),
    "google": prefixed(["AIza"], body(ALNUM + "_-", 35, 35)),
    "jwt": st.builds(lambda a, b, c: f"eyJ{a}.eyJ{b}.{c}", jwt_part, jwt_part, jwt_part),
    "gitlab": prefixed(["glpat-"], body(ALNUM + "_-", 20)),
    "hugging face": prefixed(["hf_"], body(ALNUM, 30)),
    "slack": prefixed([f"xox{c}-" for c in "abposre"], body(ALNUM + "-", 10)),
    "aws": prefixed(["AKIA", "ASIA"], body(string.ascii_uppercase + string.digits, 16, 16)),
    "pem": st.builds(
        lambda kind, lines: (
            f"-----BEGIN {kind}PRIVATE KEY-----\n"
            + "\n".join(lines)
            + f"\n-----END {kind}PRIVATE KEY-----"
        ),
        pem_kind,
        pem_lines,
    ),
}
any_key = st.one_of(*KEYS.values())
# Arbitrary text on either side, kept off the key by a separator.
side = st.text(max_size=80)
separator = st.sampled_from([" ", "\n", "\t", ", ", " (", ") ", '"', "'", ": "])


@given(side, separator, any_key, separator, side)
@example("token=", " ", "ghp_" + "A" * 20, " ", "")
def test_every_known_key_is_found_and_redacted(
    left: str, before: str, key: str, after: str, right: str
) -> None:
    text = left + before + key + after + right
    assert looks_secret(text)
    assert contains_secret(text, ())
    once = redact(text, ())
    assert key not in once
    assert "[redacted]" in once


@given(side, side)
def test_redaction_is_idempotent_and_leaves_no_key(left: str, right: str) -> None:
    once = redact(left + right, ())
    assert redact(once, ()) == once


@given(side, separator, any_key, separator, side)
def test_redacting_twice_changes_nothing(
    left: str, before: str, key: str, after: str, right: str
) -> None:
    once = redact(left + before + key + after + right, ())
    assert redact(once, ()) == once


# An environment secret is matched as a literal. Eight or more characters, not all
# digits (a bare number is ignored on purpose, see secrets.py).
literal = st.text(alphabet=ALNUM + "!#%&*+-./:=?@^_~[]", min_size=8, max_size=40).filter(
    lambda value: not value.isdigit()
)


@given(side, literal, side)
# Found while writing this property (random search did not reach it): the marker
# plus the text after it spelled the secret again.
@example("", "d]abcdef", "abcdef")
@example("[red", "acted]x1", "acted]x1")
def test_an_environment_secret_never_survives_redaction(left: str, secret: str, right: str) -> None:
    text = left + secret + right
    once = redact(text, (secret,))
    assert secret not in once
    assert contains_secret(text, (secret,))
    assert redact(once, (secret,)) == once


# Benign classes. None of these may be flagged or changed.
word = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=14)
titled = word.map(str.capitalize)
hexes = st.binary(min_size=1, max_size=64).map(bytes.hex)
upper_hex = hexes.map(str.upper)
uuids = st.uuids().map(str)
numbers = st.integers(0, 10**12).map(str)
segment = st.one_of(word, hexes, uuids, numbers, st.sampled_from(["src", "v1", "T", "_", "."]))
paths = st.builds(
    lambda head, parts, tail: head + "/".join(parts) + tail,
    st.sampled_from(["/", "./", "../", "~/", ""]),
    st.lists(segment, min_size=1, max_size=8),
    st.sampled_from(["", "/", ".py", ".txt"]),
)
benign_token = st.one_of(word, titled, hexes, upper_hex, uuids, numbers, paths)
benign = st.lists(benign_token, max_size=20).flatmap(
    lambda tokens: st.lists(
        st.sampled_from([" ", ", ", ". ", "\n"]), min_size=len(tokens), max_size=len(tokens)
    ).map(lambda gaps: "".join(token + gap for token, gap in zip(tokens, gaps, strict=True)))
)


@given(benign)
@example("da39a3ee5e6b4b0d3255bfef95601890afd80709")
@example("/private/var/folders/x8/7k2c9m3n1p0q5r6s8t4v2w1y0000gn/T/pytest-of-runner")
@example(str(uuid.UUID(int=0)) + " " + "F" * 64)
def test_benign_text_is_never_flagged(text: str) -> None:
    assert not looks_secret(text)
    assert redact(text, ()) == text


def _sample(rng: random.Random, kinds: int = 6) -> str:
    kind = rng.randrange(kinds)
    if kind == 0:
        return "".join(rng.choices(string.ascii_lowercase, k=rng.randint(1, 14)))
    if kind == 1:
        return rng.randbytes(rng.randint(1, 64)).hex()
    if kind == 2:
        return str(uuid.UUID(int=rng.getrandbits(128)))
    if kind == 3:
        return str(rng.randrange(10**12))
    if kind == 4:
        return rng.randbytes(20).hex().upper()
    return "/" + "/".join(_sample(rng, 5) for _ in range(rng.randint(1, 6)))


def test_the_false_positive_rate_on_benign_text_is_zero() -> None:
    rng = random.Random(20261018)
    samples = [" ".join(_sample(rng) for _ in range(rng.randint(1, 12))) for _ in range(5000)]
    flagged = [sample for sample in samples if looks_secret(sample)]
    rate = len(flagged) / len(samples)
    assert rate == 0, f"false-positive rate {rate:.4%}: {flagged[:3]}"

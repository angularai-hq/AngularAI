"""What a paid step costs: token usage from each response, priced per 1M tokens.

Prices are OpenAI's standard tier from platform.openai.com/docs/pricing, read on
2026-10-21. A price can be given on the command line for any other model.
"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Usage:
    """Tokens a provider billed: input (cached included), cached input, output."""

    input: int = 0
    cached: int = 0
    output: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input + other.input, self.cached + other.cached, self.output + other.output
        )

    def __sub__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input - other.input, self.cached - other.cached, self.output - other.output
        )


@dataclass(frozen=True)
class Price:
    """US dollars per 1M tokens."""

    input: float
    output: float
    cached: float

    def cost(self, usage: Usage) -> float:
        fresh = max(0, usage.input - usage.cached)
        total = fresh * self.input + usage.cached * self.cached + usage.output * self.output
        return total / 1_000_000


# Standard tier, per 1M tokens, 2026-10-21.
PRICES: Mapping[str, Price] = {
    "gpt-5.4-mini": Price(input=0.75, output=4.50, cached=0.075),
    "gpt-5.4-nano": Price(input=0.20, output=1.25, cached=0.02),
    "gpt-5.4": Price(input=2.50, output=15.00, cached=0.25),
}


def price_for(model: str) -> Price | None:
    return PRICES.get(model.strip())


def parse_price(text: str) -> Price:
    """'input,output,cached' in dollars per 1M tokens."""

    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("A price is input,output,cached in dollars per 1M tokens.")
    values = [float(part) for part in parts]
    if any(value < 0 or value != value for value in values):
        raise ValueError("A price cannot be negative.")
    return Price(*values)


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def usage_of(payload: Mapping[str, object]) -> Usage:
    """The usage block of an OpenAI or Anthropic response. Missing counts are 0."""

    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return Usage()
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, Mapping) else 0
        return Usage(
            _count(usage.get("prompt_tokens")),
            _count(cached),
            _count(usage.get("completion_tokens")),
        )
    read = _count(usage.get("cache_read_input_tokens"))
    written = _count(usage.get("cache_creation_input_tokens"))
    return Usage(
        _count(usage.get("input_tokens")) + read + written, read, _count(usage.get("output_tokens"))
    )

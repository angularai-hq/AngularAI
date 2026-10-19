from collections.abc import Sequence

CONCEPTS = {
    "color": {"blue", "colour", "color", "shade", "red"},
    "food": {"catering", "lunch", "noon", "food"},
}


def concept_vector(text: str) -> list[float]:
    words = {word.strip("?.,!").lower() for word in text.split()}
    return [float(len(words & group)) for group in CONCEPTS.values()] + [0.1]


class FakeEmbedder:
    """Vectors by shared concept. No key, no network."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.sent: list[list[str]] = []
        self.fail = fail

    @property
    def model(self) -> str:
        return "fake-1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.sent.append(list(texts))
        if self.fail is not None:
            raise self.fail
        return [concept_vector(text) for text in texts]

import re

_WORD = re.compile(r"[a-z0-9]+")

STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "i",
        "you",
        "we",
        "my",
        "me",
        "do",
        "did",
        "does",
        "what",
        "where",
        "when",
        "who",
        "how",
        "from",
        "at",
        "as",
        "if",
        "so",
        "but",
        "not",
        "your",
        "our",
        "am",
    }
)


def forms(word: str) -> set[str]:
    found = {word}
    if len(word) > 4 and word.endswith("ies"):
        found.add(word[:-3] + "y")
    if len(word) > 4 and word.endswith("es"):
        found.add(word[:-2])
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        found.add(word[:-1])
    if len(word) > 5 and word.endswith("ing"):
        found.add(word[:-3])
        found.add(word[:-3] + "e")
    if len(word) > 4 and word.endswith("ed"):
        found.add(word[:-2])
        found.add(word[:-1])
    return found


def tokens(text: str) -> set[str]:
    found: set[str] = set()
    for word in _WORD.findall(text.lower()):
        if word in STOP or len(word) < 2:
            continue
        found.update(forms(word))
    return found


def score(query: str, document: str) -> float:
    query_tokens = tokens(query)
    if not query_tokens:
        return 0.0
    document_tokens = tokens(document)
    if not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)

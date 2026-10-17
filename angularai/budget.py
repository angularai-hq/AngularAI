def estimate_tokens(text: str) -> int:
    """Membrane size. Four characters to one token, rounded up.

    This is the budget AngularAI enforces. It is not the provider's tokenizer.
    Staying under this estimate keeps every call inside a fixed window.
    """

    if not text:
        return 0
    return (len(text) + 3) // 4

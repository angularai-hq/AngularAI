import re

_SECRET = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{8,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{8,}|ghp_[A-Za-z0-9]{20,})"
)


def looks_secret(text: str) -> bool:
    return _SECRET.search(text) is not None


def redact(text: str, secrets: tuple[str, ...]) -> str:
    redacted = text
    for secret in secrets:
        if len(secret) >= 8:
            redacted = redacted.replace(secret, "[redacted]")
    return _SECRET.sub("[redacted]", redacted)


def contains_secret(text: str, secrets: tuple[str, ...]) -> bool:
    if looks_secret(text):
        return True
    return any(len(secret) >= 8 and secret in text for secret in secrets)

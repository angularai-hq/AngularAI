from urllib.parse import urlparse


def validate_base_url(value: str) -> str:
    raw = value.strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("The base URL must start with http:// or https://.")
    if parsed.username or parsed.password:
        raise ValueError("Put the key in the environment, not in the URL.")
    if any(character.isspace() for character in raw):
        raise ValueError("The base URL cannot contain spaces.")
    return raw

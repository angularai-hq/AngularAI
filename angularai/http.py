import json
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from angularai.errors import SubstrateUnavailable

TIMEOUT_SECONDS = 90.0


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> Request | None:
        return None


_OPENER = build_opener(_NoRedirect())


def http_json(
    url: str,
    headers: dict[str, str],
    payload: Mapping[str, object],
    timeout: float = TIMEOUT_SECONDS,
) -> tuple[int, bytes] | tuple[int, bytes, Mapping[str, str]]:
    """POST JSON. A success is (status, body). An HTTP error also carries its headers."""

    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers=headers, method="POST")
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            status = response.status
            body = response.read()
    except HTTPError as exc:
        raw = exc.read()
        found = {name.lower(): value for name, value in exc.headers.items()} if exc.headers else {}
        return exc.code, raw if isinstance(raw, bytes) else b"", found
    except TimeoutError as exc:
        raise SubstrateUnavailable("The substrate timed out.") from exc
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise SubstrateUnavailable("The substrate timed out.") from exc
        raise SubstrateUnavailable("The substrate is unreachable.") from exc
    if not isinstance(status, int):
        raise SubstrateUnavailable("The substrate returned an unreadable status.")
    if not isinstance(body, bytes):
        body = bytes(body)
    return status, body

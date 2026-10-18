"""One JSON POST, with a connect timeout and a read timeout.

http.client is used, not urllib, so the two timeouts are separate: the connect
timeout covers TCP and TLS, the read timeout covers every wait for bytes after.
Redirects are never followed. HTTPS_PROXY / HTTP_PROXY / NO_PROXY are honored.
"""

import base64
import http.client
import json
import ssl
from collections.abc import Callable, Mapping
from urllib.parse import SplitResult, unquote, urlsplit
from urllib.request import getproxies, proxy_bypass

from angularai.errors import BadResponse, SubstrateUnavailable, Timeout, Unreachable

CONNECT_SECONDS = 10.0
TIMEOUT_SECONDS = 90.0

Answer = tuple[int, bytes] | tuple[int, bytes, Mapping[str, str]]
Connection = http.client.HTTPConnection
# (target URL, connect timeout) -> (connection, request target, extra request headers)
Route = tuple[Connection, str, dict[str, str]]
Connector = Callable[[SplitResult, float], Route]


def connection(url: SplitResult, connect_timeout: float) -> Route:
    """A connection for url, through the environment's proxy when there is one."""

    target = url.path or "/"
    if url.query:
        target += f"?{url.query}"
    host = url.hostname or ""
    secure = url.scheme == "https"
    port = url.port or (443 if secure else 80)
    proxy = getproxies().get(url.scheme)
    if proxy and not proxy_bypass(host):
        through = urlsplit(proxy if "://" in proxy else f"http://{proxy}")
        headers: dict[str, str] = {}
        if through.username is not None:
            pair = f"{unquote(through.username)}:{unquote(through.password or '')}"
            headers["Proxy-Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode()
        proxy_host = through.hostname or ""
        proxy_port = through.port or (443 if through.scheme == "https" else 80)
        if secure:
            tunnel = http.client.HTTPSConnection(
                proxy_host,
                proxy_port,
                timeout=connect_timeout,
                context=ssl.create_default_context(),
            )
            tunnel.set_tunnel(host, port, headers=headers)
            return tunnel, target, {}
        plain = http.client.HTTPConnection(proxy_host, proxy_port, timeout=connect_timeout)
        return plain, url.geturl(), headers
    if secure:
        context = ssl.create_default_context()
        secured = http.client.HTTPSConnection(host, port, timeout=connect_timeout, context=context)
        return secured, target, {}
    return http.client.HTTPConnection(host, port, timeout=connect_timeout), target, {}


def http_json(
    url: str,
    headers: dict[str, str],
    payload: Mapping[str, object],
    timeout: float = TIMEOUT_SECONDS,
    *,
    connect_timeout: float = CONNECT_SECONDS,
    connect: Connector = connection,
) -> Answer:
    """POST JSON. A success is (status, body). An HTTP error also carries its headers.

    timeout is the read timeout. A timeout raises Timeout, a failed or reset
    connection Unreachable, and a body cut short BadResponse.
    """

    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise Unreachable("The substrate URL is not http or https.")
    data = json.dumps(payload).encode("utf-8")
    conn, target, extra = connect(parts, connect_timeout)
    try:
        conn.connect()
        if conn.sock is not None:
            conn.sock.settimeout(timeout)
        conn.request("POST", target, body=data, headers={**headers, **extra})
        response = conn.getresponse()
        status = response.status
        found = {name.lower(): value for name, value in response.getheaders()}
        body = response.read()
    except TimeoutError as exc:
        raise Timeout("The substrate timed out.") from exc
    except http.client.IncompleteRead as exc:
        raise BadResponse("The substrate sent a body that was cut short.") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise Unreachable("The substrate is unreachable.") from exc
    finally:
        conn.close()
    if not isinstance(status, int):
        raise SubstrateUnavailable("The substrate returned an unreadable status.")
    if not isinstance(body, bytes):
        body = bytes(body)
    if 200 <= status < 300:
        return status, body
    return status, body, found

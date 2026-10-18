"""Loopback token, Host, and same-origin checks.

The server binds 127.0.0.1 only, but any web page the user visits can still
send requests to localhost. Three checks stop that:

* **Host header** must be a loopback name, which defeats DNS rebinding (an
  attacker's domain resolving to 127.0.0.1 still sends its own Host).
* **Same origin**: cookies are scoped by host, *not* port, so SameSite=Strict
  alone would let a page on another localhost port (any dev server) ride the
  session. ``/api`` therefore rejects requests whose ``Sec-Fetch-Site`` is
  ``cross-site``/``same-site`` or whose ``Origin`` is not exactly this server.
* **Token**: every ``/api`` request needs the per-run secret, via the
  ``X-LCTIMS-Token`` header, ``Authorization: Bearer``, or the session cookie.
  Opening ``/?token=...`` (the URL ``lctimspy serve`` prints) sets that cookie,
  so ``fetch`` and ``EventSource`` -- which cannot set headers -- just work.
  The cookie name includes the port, so two instances never share or clobber
  one. The query-string token is accepted only for that first page load, never
  on ``/api``, so it does not end up in API logs or referrers.
"""

from __future__ import annotations

import secrets

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

COOKIE_PREFIX = "lctims_token"
HEADER_NAME = "x-lctims-token"
# Fetch-metadata values that mean "the request came from this very origin" or
# "the user typed/bookmarked it". Anything else on /api is refused.
ALLOWED_FETCH_SITES = frozenset({"same-origin", "none"})


def _split_host(host_header: str) -> tuple[str, str]:
    if host_header.startswith("["):
        name, _, rest = host_header.partition("]")
        return name + "]", rest.lstrip(":")
    if ":" in host_header:
        name, port = host_header.rsplit(":", 1)
        return name, port
    return host_header, ""


def cookie_name(request: Request) -> str:
    port = _split_host(request.headers.get("host", ""))[1]
    return f"{COOKIE_PREFIX}_{port}" if port else COOKIE_PREFIX


def request_token(request: Request) -> str | None:
    """The token sent by headers or the session cookie (never the query string)."""
    token = request.headers.get(HEADER_NAME)
    if token:
        return token
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.cookies.get(cookie_name(request))


def _cross_origin(request: Request) -> str | None:
    """Why this request is not from our own origin, or None if it is."""
    site = request.headers.get("sec-fetch-site")
    if site is not None and site not in ALLOWED_FETCH_SITES:
        return f"Sec-Fetch-Site is {site!r}"
    origin = request.headers.get("origin")
    if origin is not None and origin != "null":
        expected = f"http://{request.headers.get('host', '')}"
        if origin != expected:
            return f"Origin {origin!r} is not {expected!r}"
    elif origin == "null":
        return "opaque origin"
    return None


class SecurityMiddleware:
    """Pure ASGI middleware: unlike BaseHTTPMiddleware it never buffers or wraps
    the response, which matters for the long-lived SSE stream."""

    def __init__(self, app, token: str | None, allowed_hosts: tuple[str, ...]):
        self.app = app
        self.token = token
        self.allowed_hosts = set(allowed_hosts)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        response = self._check(request)
        if response is not None:
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def _check(self, request: Request):
        host = _split_host(request.headers.get("host", ""))[0]
        if host not in self.allowed_hosts:
            return JSONResponse({"error": "forbidden", "detail": f"host {host!r} not allowed"},
                                status_code=403)
        path = request.url.path
        is_api = path.startswith("/api")
        if is_api:
            reason = _cross_origin(request)
            if reason is not None:
                return JSONResponse({"error": "forbidden",
                                     "detail": f"cross-origin request refused: {reason}"},
                                    status_code=403)
        if self.token is None:
            return None

        # Opening the printed URL: set the cookie and drop the token from the URL.
        # Checked on its own so a stale cookie from an earlier run is replaced.
        if not is_api and self._matches(request.query_params.get("token")):
            response = RedirectResponse(str(request.url.remove_query_params("token")),
                                        status_code=303)
            response.set_cookie(cookie_name(request), self.token, httponly=True,
                                samesite="strict")
            return response

        if is_api and path != "/api/health" and not self._matches(request_token(request)):
            return JSONResponse({"error": "unauthorized",
                                 "detail": "missing or invalid token; open the URL that "
                                           "`lctimspy serve` printed"}, status_code=401)
        return None

    def _matches(self, supplied: str | None) -> bool:
        return supplied is not None and secrets.compare_digest(
            supplied.encode("utf-8"), self.token.encode("utf-8"))

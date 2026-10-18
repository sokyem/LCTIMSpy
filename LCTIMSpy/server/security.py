"""Loopback token and Host checks.

The server binds 127.0.0.1 only, but any web page the user visits can still
send requests to localhost. Two checks stop that:

* **Host header** must be a loopback name, which defeats DNS rebinding (an
  attacker's domain resolving to 127.0.0.1 still sends its own Host).
* **Token**: every ``/api`` request needs the per-run secret, via the
  ``X-LCTIMS-Token`` header, ``Authorization: Bearer``, or the session cookie.
  Opening ``/?token=...`` (the URL ``lctimspy serve`` prints) sets that cookie,
  so ``fetch`` and ``EventSource`` -- which cannot set headers -- just work.
"""

from __future__ import annotations

import secrets

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

COOKIE_NAME = "lctims_token"
HEADER_NAME = "x-lctims-token"


def _host_name(host_header: str) -> str:
    if host_header.startswith("["):
        return host_header.split("]")[0] + "]"
    return host_header.rsplit(":", 1)[0] if ":" in host_header else host_header


def request_token(request: Request) -> str | None:
    token = request.headers.get(HEADER_NAME)
    if token:
        return token
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.cookies.get(COOKIE_NAME) or request.query_params.get("token")


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
        host = _host_name(request.headers.get("host", ""))
        if host not in self.allowed_hosts:
            return JSONResponse({"error": "forbidden", "detail": f"host {host!r} not allowed"},
                                status_code=403)
        if self.token is None:
            return None
        path = request.url.path
        # Opening the printed URL: set the cookie and drop the token from the URL.
        # Checked on its own so a stale cookie from an earlier run is replaced.
        if not path.startswith("/api") and self._matches(request.query_params.get("token")):
            response = RedirectResponse(str(request.url.remove_query_params("token")),
                                        status_code=303)
            response.set_cookie(COOKIE_NAME, self.token, httponly=True, samesite="strict")
            return response

        if (path.startswith("/api") and path != "/api/health"
                and not self._matches(request_token(request))):
            return JSONResponse({"error": "unauthorized",
                                 "detail": "missing or invalid token; open the URL that "
                                           "`lctimspy serve` printed"}, status_code=401)
        return None

    def _matches(self, supplied: str | None) -> bool:
        return supplied is not None and secrets.compare_digest(
            supplied.encode("utf-8"), self.token.encode("utf-8"))

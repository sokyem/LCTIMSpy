"""Bind and run the server; shared by ``lctimspy serve`` and the desktop shell."""

from __future__ import annotations

import socket

LOOPBACK = "127.0.0.1"


def bind_loopback(port: int = 0) -> socket.socket:
    """Bind 127.0.0.1:``port`` (0 = any free port) and keep the socket.

    The bound socket is handed to uvicorn as-is. Picking a free port, closing
    it, and letting uvicorn re-bind would race with any other process grabbing
    the port in between. Never 0.0.0.0: that would put an unauthenticated
    filesystem browser on the LAN if the token were ever disabled.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):            # Windows: no port hijacking
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    sock.bind((LOOPBACK, port))
    sock.listen(128)
    return sock


def make_server(app):
    """A uvicorn Server; call ``server.run(sockets=[sock])`` to serve on the bound socket."""
    import uvicorn

    config = uvicorn.Config(app, log_level="warning", lifespan="on", access_log=False)
    server = uvicorn.Server(config)
    return server


def url_for(sock: socket.socket, token: str | None) -> str:
    port = sock.getsockname()[1]
    return f"http://{LOOPBACK}:{port}/" + (f"?token={token}" if token else "")

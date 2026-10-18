"""Server configuration."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from pathlib import Path

from ..core.io.paths import APP_NAME, cache_root


def _data_dir() -> Path:
    import platformdirs

    return Path(platformdirs.user_data_dir(APP_NAME, APP_NAME))


@dataclass
class ServerSettings:
    cache_dir: Path = field(default_factory=cache_root)
    state_dir: Path = field(default_factory=lambda: _data_dir() / "server")
    scratch_dir: Path = field(default_factory=lambda: _data_dir() / "scratch")
    # Per-run secret; None disables the check (tests, or a deliberate --no-token).
    token: str | None = field(default_factory=lambda: secrets.token_urlsafe(24))
    # The filesystem browser is the only endpoint that speaks absolute paths.
    # Turning it off is the whole change needed for a shared deployment.
    browse_enabled: bool = True
    max_concurrent_jobs: int = 2
    cancel_grace_s: float = 10.0
    # Hosts accepted in the Host header (DNS-rebinding protection). The port is
    # not checked; the token is the actual credential.
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "[::1]", "testserver")
    # Merged into every extraction worker spec. Tests use it to inject a fake
    # frame source (the worker honours that only when LCTIMSPY_TESTING=1).
    worker_spec_extra: dict = field(default_factory=dict)

"""Print the API's OpenAPI schema: ``python -m LCTIMSpy.server.openapi > openapi.json``.

The frontend generates its TypeScript types from this (``npm run gen:api``), so
the client cannot drift from the server's pydantic models.
"""

import json
import sys
import tempfile
from pathlib import Path


def main() -> None:
    from .app import create_app
    from .settings import ServerSettings

    tmp = Path(tempfile.mkdtemp())
    app = create_app(ServerSettings(cache_dir=tmp / "c", state_dir=tmp / "s",
                                    scratch_dir=tmp / "x", token=None))
    sys.stdout.write(json.dumps(app.openapi(), indent=1))


if __name__ == "__main__":
    main()

"""Serve the GUI over three synthetic, already-extracted replicates.

    python -m tests.e2e.demo_server [--port 8799] [--token TOKEN] [--root DIR]

Used by the Playwright tests, and handy for trying the interface without real
timsTOF data or the Bruker SDK. Each replicate holds an isomer pair for two
fragment ions (m/z 500.25 and 700.40) eluting at 0.40 and 0.80 min.

For the Discovery screen there is also a synthetic DDA-PASEF run with known
isomer pairs (``PASEF_isomers.d``) and a PEAKS export of its PSMs
(``PASEF_isomers.psms.csv``, next to it in the data folder).
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from LCTIMSpy.core.io.cache import Cache
from LCTIMSpy.core.io.extract import extract_to_cache
from LCTIMSpy.discovery.synthetic import Scenario, SyntheticFrameSource, write_synthetic_d
from LCTIMSpy.server.app import create_app
from LCTIMSpy.server.registry import DatasetRegistry
from LCTIMSpy.server.runner import bind_loopback, make_server, url_for
from LCTIMSpy.server.settings import ServerSettings
from tests.api.test_discovery_api import SPECIES, write_peaks_csv
from tests.conftest import FakeFrameSource, make_fake_d

DISCOVERY_RUN = "PASEF_isomers.d"


def build(root: Path) -> ServerSettings:
    settings = ServerSettings(cache_dir=root / "cache", state_dir=root / "state",
                              scratch_dir=root / "scratch")
    cache = Cache(settings.cache_dir)
    registry = DatasetRegistry(settings.state_dir, cache)
    for i in range(3):
        d = make_fake_d(root / "data", f"RH59{56 + i}.d")
        extract_to_cache(d, cache=cache, frame_source=FakeFrameSource(seed=i))
        registry.add(str(d), group="isoAsp")
    scenario = Scenario(SPECIES, n_frames=500, seconds_per_frame=1.0, msms_every=3,
                        isolation_scans=30)
    d = write_synthetic_d(root / "data", DISCOVERY_RUN, scenario)
    extract_to_cache(d, cache=cache, frame_source=SyntheticFrameSource(scenario))
    registry.add(str(d), group="discovery")
    write_peaks_csv(root / "data" / "PASEF_isomers.psms.csv", SPECIES, run=DISCOVERY_RUN)
    # One more, deliberately not extracted, to show the "Extract" state.
    registry.add(str(make_fake_d(root / "data", "RH5961.d")))
    return settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--token", default="demo")
    ap.add_argument("--root", default=None, help="state folder (default: a fresh temp dir)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(tempfile.mkdtemp(prefix="lctims-demo-"))
    if args.root and root.exists():
        shutil.rmtree(root)
    settings = build(root)
    settings.token = args.token or None
    sock = bind_loopback(args.port)
    print(f"LCTIMSpy demo at {url_for(sock, settings.token)}", flush=True)
    make_server(create_app(settings)).run(sockets=[sock])


if __name__ == "__main__":
    main()

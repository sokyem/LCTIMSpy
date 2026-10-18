"""Shared test helpers.

Until the lazy-SDK work lands (Phase 1), ``import LCTIMSpy`` executes
``LCTIMSpy/lctims.py`` which calls ``init_tdf_sdk_api()`` at module scope, so the
package cannot be imported on a machine without the Bruker SDK -- including CI.

``load_package_module`` sidesteps the package ``__init__`` and loads a single
module straight from its file, which lets us test the SDK-independent logic
today. Once ``__init__`` is lazy these tests can switch to a plain import.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "LCTIMSpy"


def load_package_module(module_name: str):
    """Load ``LCTIMSpy/<module_name>.py`` without triggering the package __init__."""
    path = PACKAGE_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_lctimspy_direct_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def fragments():
    return load_package_module("get_fragmentMz")


PROTON = 1.00727646677

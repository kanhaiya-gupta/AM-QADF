"""
Pytest conftest for am_qadf_native bridge tests.

Prepends the built native module directory to sys.path when it exists
(build/src/am_qadf_native) so bridge tests use the module with numpy_to_openvdb
and openvdb_to_numpy without requiring PYTHONPATH to be set manually.
"""

import sys
from pathlib import Path

# Project root: tests/unit/python/bridge -> 5 parents (bridge->python->unit->tests->root)
_project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
_build_native = _project_root / "build" / "src" / "am_qadf_native"
if _build_native.is_dir():
    _path = str(_build_native)
    if _path not in sys.path:
        sys.path.insert(0, _path)

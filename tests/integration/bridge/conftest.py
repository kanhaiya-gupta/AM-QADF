"""Pytest fixtures for bridge tests: skip when am_qadf_native is not available."""

import pytest


@pytest.fixture(scope="session")
def native_module():
    """Import am_qadf_native; skip all bridge tests if not available."""
    try:
        import am_qadf_native as m
        return m
    except ImportError:
        pytest.skip("am_qadf_native not built or not on PYTHONPATH")

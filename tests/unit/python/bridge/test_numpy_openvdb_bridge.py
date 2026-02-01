"""
Unit tests for am_qadf_native NumPy–OpenVDB bridge (numpy_to_openvdb, openvdb_to_numpy).

Tests the Python-only bridge in am_qadf_native; requires the built native module
with the bridge (build with CMake, then set PYTHONPATH to the directory containing
am_qadf_native*.so, e.g. build/src/am_qadf_native).
Aligned with src/am_qadf_native/include/am_qadf_native/bridge/numpy_openvdb_bridge.hpp.
"""

import numpy as np
import pytest

pytest.importorskip("am_qadf_native")

try:
    from am_qadf_native import numpy_to_openvdb, openvdb_to_numpy
except ImportError as e:
    pytest.skip(
        "am_qadf_native does not expose numpy_to_openvdb/openvdb_to_numpy. "
        "Use the built module: set PYTHONPATH to the dir with am_qadf_native*.so "
        "(e.g. build/src/am_qadf_native), then run pytest.",
        allow_module_level=True,
    )


class TestNumpyToOpenVDB:
    """numpy_to_openvdb: NumPy 3D array -> OpenVDB FloatGrid."""

    def test_3d_array_creates_non_null_grid(self):
        arr = np.zeros((4, 4, 4), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        grid = numpy_to_openvdb(arr, 0.1)
        assert grid is not None

    def test_non_zero_values_stored(self):
        arr = np.zeros((2, 2, 2), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        arr[0, 0, 1] = 2.0
        arr[1, 1, 1] = 3.0
        grid = numpy_to_openvdb(arr, 1.0)
        assert grid is not None
        back = openvdb_to_numpy(grid)
        assert back.ndim == 3
        assert back.shape[0] >= 2 and back.shape[1] >= 2 and back.shape[2] >= 2
        assert back[0, 0, 0] == pytest.approx(1.0)
        assert back[0, 0, 1] == pytest.approx(2.0)
        assert back[1, 1, 1] == pytest.approx(3.0)

    def test_invalid_ndim_1d_raises(self):
        arr = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError, match="3D"):
            numpy_to_openvdb(arr, 0.1)

    def test_invalid_ndim_2d_raises(self):
        arr = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="3D"):
            numpy_to_openvdb(arr, 0.1)

    def test_voxel_size_accepts_different_sizes(self):
        arr = np.zeros((2, 2, 2), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        grid_small = numpy_to_openvdb(arr, 0.1)
        grid_large = numpy_to_openvdb(arr, 1.0)
        assert grid_small is not None
        assert grid_large is not None
        # Round-trip still works for both
        np.testing.assert_array_almost_equal(openvdb_to_numpy(grid_small)[0, 0, 0], 1.0)
        np.testing.assert_array_almost_equal(openvdb_to_numpy(grid_large)[0, 0, 0], 1.0)


class TestOpenVDBToNumPy:
    """openvdb_to_numpy: OpenVDB FloatGrid -> NumPy array."""

    def test_null_grid_returns_empty_shaped_array(self):
        result = openvdb_to_numpy(None)
        assert result.ndim == 3
        assert result.shape == (0, 0, 0)

    def test_empty_grid_returns_zero_sized(self):
        # Create empty grid via numpy_to_openvdb with (0,0,0) array
        arr = np.zeros((0, 0, 0), dtype=np.float32)
        grid = numpy_to_openvdb(arr, 0.1)
        assert grid is not None
        result = openvdb_to_numpy(grid)
        assert result.ndim == 3
        assert result.shape == (0, 0, 0)


class TestRoundTrip:
    """numpy_to_openvdb then openvdb_to_numpy round-trip."""

    def test_round_trip_preserves_values(self):
        arr = np.arange(27, dtype=np.float32).reshape(3, 3, 3) + 1.0
        grid = numpy_to_openvdb(arr, 0.5)
        assert grid is not None
        back = openvdb_to_numpy(grid)
        assert back.ndim == 3
        assert back.shape[0] >= 1 and back.shape[1] >= 1 and back.shape[2] >= 1
        # Values in round-trip region should match
        for z in range(min(3, back.shape[0])):
            for y in range(min(3, back.shape[1])):
                for x in range(min(3, back.shape[2])):
                    expected = 1.0 + (z * 9 + y * 3 + x)
                    assert back[z, y, x] == pytest.approx(expected, rel=1e-5)

    def test_round_trip_small_sparse(self):
        arr = np.zeros((2, 2, 2), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        arr[1, 1, 1] = 2.0
        grid = numpy_to_openvdb(arr, 1.0)
        back = openvdb_to_numpy(grid)
        assert back.ndim == 3
        assert back.shape[0] >= 2 and back.shape[1] >= 2 and back.shape[2] >= 2
        assert back[0, 0, 0] == pytest.approx(1.0)
        assert back[1, 1, 1] == pytest.approx(2.0)

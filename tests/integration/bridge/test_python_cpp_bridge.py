"""
Bridge tests: basic Python–C++ module import and submodule presence.

Aligned with docs/Tests/Test_New_Plans.md (tests/integration/bridge/).
"""

import pytest


@pytest.mark.integration
@pytest.mark.bridge
class TestPythonCppBridge:
    """Basic am_qadf_native module and submodule availability."""

    def test_import_am_qadf_native(self):
        """Module can be imported when built and on PYTHONPATH."""
        try:
            import am_qadf_native
        except ImportError:
            pytest.skip("am_qadf_native not built or not on PYTHONPATH")
        assert am_qadf_native is not None

    def test_module_has_io_submodule(self, native_module):
        """Module exposes io submodule (OpenVDBReader, VDBWriter, etc.)."""
        assert hasattr(native_module, "io")
        io_m = native_module.io
        assert hasattr(io_m, "OpenVDBReader")
        assert hasattr(io_m, "VDBWriter")

    def test_module_has_signal_mapping_submodule(self, native_module):
        """Module exposes signal_mapping (NearestNeighborMapper, Point, etc.)."""
        assert hasattr(native_module, "signal_mapping")
        sm = native_module.signal_mapping
        assert hasattr(sm, "NearestNeighborMapper")
        assert hasattr(sm, "Point")

    def test_module_has_voxelization_classes(self, native_module):
        """Module exposes voxelization (UniformVoxelGrid, VoxelGridFactory)."""
        assert hasattr(native_module, "UniformVoxelGrid")
        assert hasattr(native_module, "VoxelGridFactory")

    def test_module_has_fusion(self, native_module):
        """Module exposes fusion submodule and GridFusion."""
        assert hasattr(native_module, "fusion")
        assert hasattr(native_module.fusion, "GridFusion")

    def test_module_has_synchronization(self, native_module):
        """Module exposes SpatialAlignment, GridSynchronizer."""
        assert hasattr(native_module, "SpatialAlignment")
        assert hasattr(native_module, "GridSynchronizer")

    def test_module_has_query_result(self, native_module):
        """Module exposes QueryResult and MongoDBQueryClient."""
        assert hasattr(native_module, "QueryResult")
        assert hasattr(native_module, "MongoDBQueryClient")

    def test_module_has_correction(self, native_module):
        """Module exposes correction submodule with SignalNoiseReduction, Calibration."""
        assert hasattr(native_module, "correction")
        assert hasattr(native_module.correction, "SignalNoiseReduction")
        assert hasattr(native_module.correction, "Calibration")

    def test_module_has_processing(self, native_module):
        """Module exposes processing submodule with SignalProcessing, SignalGeneration."""
        assert hasattr(native_module, "processing")
        assert hasattr(native_module.processing, "SignalProcessing")
        assert hasattr(native_module.processing, "SignalGeneration")

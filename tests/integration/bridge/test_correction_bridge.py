"""
Bridge tests: correction (SignalNoiseReduction, Calibration, Validation) via am_qadf_native.

Aligned with docs/Tests/Test_New_Plans.md (tests/integration/bridge/).
"""

import pytest


@pytest.mark.integration
@pytest.mark.bridge
class TestCorrectionBridge:
    """Python → C++ correction API."""

    def test_signal_noise_reduction_reduce_noise(self, native_module):
        """SignalNoiseReduction.reduce_noise returns QueryResult with same length."""
        QueryResult = native_module.QueryResult
        SignalNoiseReduction = native_module.correction.SignalNoiseReduction
        raw = QueryResult()
        raw.points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        raw.values = [10.0, 12.0, 11.0]
        raw.timestamps = [0.0, 1.0, 2.0]
        raw.layers = [0, 0, 0]
        reducer = SignalNoiseReduction()
        out = reducer.reduce_noise(raw, "gaussian", 1.0)
        assert out is not None
        assert len(out.points) == len(raw.points)
        assert len(out.values) == len(raw.values)

    def test_signal_noise_reduction_apply_gaussian_filter(self, native_module):
        """SignalNoiseReduction.apply_gaussian_filter."""
        QueryResult = native_module.QueryResult
        SignalNoiseReduction = native_module.correction.SignalNoiseReduction
        raw = QueryResult()
        raw.points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        raw.values = [5.0, 15.0]
        raw.timestamps = [0.0, 1.0]
        raw.layers = [0, 0]
        reducer = SignalNoiseReduction()
        out = reducer.apply_gaussian_filter(raw, 0.5)
        assert len(out.points) == 2
        assert len(out.values) == 2

    def test_signal_noise_reduction_remove_outliers(self, native_module):
        """SignalNoiseReduction.remove_outliers."""
        QueryResult = native_module.QueryResult
        SignalNoiseReduction = native_module.correction.SignalNoiseReduction
        raw = QueryResult()
        raw.points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        raw.values = [10.0, 1000.0, 11.0]  # outlier at index 1
        raw.timestamps = [0.0, 1.0, 2.0]
        raw.layers = [0, 0, 0]
        reducer = SignalNoiseReduction()
        out = reducer.remove_outliers(raw, 3.0)
        assert len(out.points) == 3
        assert len(out.values) == 3

    def test_calibration_construct_and_load_from_file(self, native_module):
        """Calibration can be constructed; load_from_file on missing file returns empty CalibrationData."""
        Calibration = native_module.correction.Calibration
        cal = Calibration()
        assert cal is not None
        result = cal.load_from_file("_nonexistent_cal_file_12345.txt")
        assert result is not None
        assert hasattr(result, "sensor_id")

    def test_validation_validate_grid(self, native_module):
        """Validation.validate_grid accepts OpenVDB grid."""
        if not hasattr(native_module, "correction") or not hasattr(native_module.correction, "Validation"):
            pytest.skip("Validation not exposed")
        UniformVoxelGrid = native_module.UniformVoxelGrid
        Validation = native_module.correction.Validation
        grid = UniformVoxelGrid(1.0)
        grid.add_point_at_voxel(0, 0, 0, 1.0)
        val = Validation()
        result = val.validate_grid(grid.get_grid())
        assert result is not None

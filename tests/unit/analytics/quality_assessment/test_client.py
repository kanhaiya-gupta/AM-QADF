"""
Unit tests for quality assessment client (analytics).

Tests for QualityAssessmentClient.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from am_qadf.analytics.quality_assessment.client import QualityAssessmentClient
from am_qadf.analytics.quality_assessment.completeness import GapFillingStrategy


class MockVoxelData:
    """Mock voxel data object for testing."""

    def __init__(
        self,
        signals: dict,
        dims: tuple = (10, 10, 10),
        bbox_min: tuple = (0, 0, 0),
        resolution: float = 1.0,
    ):
        """Initialize with signal dictionary."""
        self._signals = signals
        self.dims = dims
        self.bbox_min = bbox_min
        self.resolution = resolution
        self.available_signals = list(signals.keys())

    def get_signal_array(self, signal_name: str, default: float = 0.0) -> np.ndarray:
        """Get signal array by name."""
        return self._signals.get(signal_name, np.full(self.dims, default))


class TestQualityAssessmentClient:
    """Test suite for QualityAssessmentClient class."""

    @pytest.fixture
    def client(self):
        """Create a QualityAssessmentClient instance."""
        return QualityAssessmentClient()

    @pytest.fixture
    def mock_voxel_data(self):
        """Create mock voxel data."""
        signals = {
            "signal1": np.ones((10, 10, 10)),
            "signal2": np.ones((10, 10, 10)) * 2.0,
        }
        return MockVoxelData(signals, dims=(10, 10, 10))

    @pytest.mark.unit
    def test_client_creation_default(self):
        """Test creating QualityAssessmentClient with default parameters."""
        client = QualityAssessmentClient()

        assert client.data_quality_analyzer is not None
        assert client.signal_quality_analyzer is not None
        assert client.alignment_analyzer is not None
        assert client.completeness_analyzer is not None

    @pytest.mark.unit
    def test_client_creation_custom(self):
        """Test creating QualityAssessmentClient with custom parameters."""
        client = QualityAssessmentClient(max_acceptable_error=0.2, noise_floor=1e-5)

        assert client.alignment_analyzer.max_acceptable_error == 0.2
        assert client.signal_quality_analyzer.noise_floor == 1e-5

    @pytest.mark.unit
    def test_assess_data_quality(self, client, mock_voxel_data):
        """Test assessing data quality."""
        result = client.assess_data_quality(mock_voxel_data, ["signal1", "signal2"])

        assert result is not None
        assert hasattr(result, "completeness")
        assert hasattr(result, "coverage_spatial")
        assert hasattr(result, "coverage_temporal")
        assert 0.0 <= result.completeness <= 1.0

    @pytest.mark.unit
    def test_assess_data_quality_with_layer_range(self, client, mock_voxel_data):
        """Test assessing data quality with layer range."""
        result = client.assess_data_quality(mock_voxel_data, ["signal1"], layer_range=(0, 5))

        assert result is not None
        assert 0.0 <= result.coverage_temporal <= 1.0

    @pytest.mark.unit
    def test_assess_signal_quality(self, client):
        """Test assessing signal quality for a single signal."""
        signal_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        result = client.assess_signal_quality("test_signal", signal_array)

        assert result is not None
        assert result.signal_name == "test_signal"
        assert result.snr_mean >= 0
        assert 0.0 <= result.confidence_mean <= 1.0
        assert 0.0 <= result.quality_score <= 1.0

    @pytest.mark.unit
    def test_assess_signal_quality_with_noise_estimate(self, client):
        """Test assessing signal quality with noise estimate."""
        signal_array = np.array([100.0, 200.0, 300.0])
        noise_estimate = np.array([10.0, 20.0, 30.0])

        result = client.assess_signal_quality("test_signal", signal_array, noise_estimate=noise_estimate)

        assert result is not None
        assert result.signal_name == "test_signal"

    @pytest.mark.unit
    def test_assess_signal_quality_no_store_maps(self, client):
        """Test assessing signal quality without storing maps."""
        signal_array = np.array([100.0, 200.0, 300.0])

        result = client.assess_signal_quality("test_signal", signal_array, store_maps=False)

        assert result.snr_map is None
        assert result.uncertainty_map is None
        assert result.confidence_map is None

    @pytest.mark.unit
    def test_assess_all_signals(self, client, mock_voxel_data):
        """Test assessing quality for all signals."""
        result = client.assess_all_signals(mock_voxel_data, ["signal1", "signal2"])

        assert isinstance(result, dict)
        assert "signal1" in result
        assert "signal2" in result
        assert result["signal1"].signal_name == "signal1"
        assert result["signal2"].signal_name == "signal2"

    @pytest.mark.unit
    def test_assess_all_signals_no_store_maps(self, client, mock_voxel_data):
        """Test assessing all signals without storing maps."""
        result = client.assess_all_signals(mock_voxel_data, ["signal1"], store_maps=False)

        assert isinstance(result, dict)
        assert result["signal1"].snr_map is None

    @pytest.mark.unit
    def test_assess_alignment_accuracy(self, client, mock_voxel_data):
        """Test assessing alignment accuracy."""
        result = client.assess_alignment_accuracy(mock_voxel_data)

        assert result is not None
        assert hasattr(result, "alignment_score")
        assert 0.0 <= result.alignment_score <= 1.0

    @pytest.mark.unit
    def test_assess_alignment_accuracy_with_transformer(self, client, mock_voxel_data):
        """Test assessing alignment accuracy with coordinate transformer."""
        transformer = Mock()
        result = client.assess_alignment_accuracy(mock_voxel_data, coordinate_transformer=transformer)

        assert result is not None

    @pytest.mark.unit
    def test_assess_completeness(self, client, mock_voxel_data):
        """Test assessing completeness."""
        result = client.assess_completeness(mock_voxel_data, ["signal1", "signal2"])

        assert result is not None
        assert hasattr(result, "completeness_ratio")
        assert 0.0 <= result.completeness_ratio <= 1.0

    @pytest.mark.unit
    def test_assess_completeness_no_store_details(self, client, mock_voxel_data):
        """Test assessing completeness without storing details."""
        result = client.assess_completeness(mock_voxel_data, ["signal1"], store_details=False)

        assert result is not None

    @pytest.mark.unit
    def test_fill_gaps(self, client):
        """Test filling gaps in signal array."""
        signal_array = np.array([1.0, 2.0, np.nan, 0.0, 5.0])

        filled = client.fill_gaps(signal_array, strategy=GapFillingStrategy.MEAN)

        assert isinstance(filled, np.ndarray)
        assert not np.isnan(filled).any()

    @pytest.mark.unit
    def test_fill_gaps_linear(self, client):
        """Test filling gaps with linear strategy."""
        signal_array = np.array([1.0, 2.0, np.nan, 0.0, 5.0])

        filled = client.fill_gaps(signal_array, strategy=GapFillingStrategy.LINEAR)

        assert isinstance(filled, np.ndarray)
        assert not np.isnan(filled).any()

    @pytest.mark.unit
    def test_comprehensive_assessment(self, client, mock_voxel_data):
        """Test comprehensive quality assessment."""
        result = client.comprehensive_assessment(mock_voxel_data, ["signal1", "signal2"])

        assert isinstance(result, dict)
        assert "data_quality" in result
        assert "signal_quality" in result
        assert "alignment_accuracy" in result
        assert "completeness" in result
        assert "summary" in result

    @pytest.mark.unit
    def test_comprehensive_assessment_with_layer_range(self, client, mock_voxel_data):
        """Test comprehensive assessment with layer range."""
        result = client.comprehensive_assessment(mock_voxel_data, ["signal1"], layer_range=(0, 5))

        assert isinstance(result, dict)
        assert "summary" in result

    @pytest.mark.unit
    def test_comprehensive_assessment_with_transformer(self, client, mock_voxel_data):
        """Test comprehensive assessment with coordinate transformer."""
        transformer = Mock()
        result = client.comprehensive_assessment(mock_voxel_data, ["signal1"], coordinate_transformer=transformer)

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_comprehensive_assessment_no_store_maps(self, client, mock_voxel_data):
        """Test comprehensive assessment without storing maps."""
        result = client.comprehensive_assessment(mock_voxel_data, ["signal1"], store_maps=False)

        assert isinstance(result, dict)
        # Signal quality maps should be None
        assert all(sq.snr_map is None for sq in result["signal_quality"].values())

    @pytest.mark.unit
    def test_generate_quality_report(self, client, mock_voxel_data):
        """Test generating quality report."""
        assessment_results = client.comprehensive_assessment(mock_voxel_data, ["signal1", "signal2"])

        report = client.generate_quality_report(assessment_results)

        assert isinstance(report, str)
        assert len(report) > 0
        assert "QUALITY ASSESSMENT REPORT" in report
        assert "SUMMARY" in report

    @pytest.mark.unit
    def test_generate_quality_report_with_output_file(self, client, mock_voxel_data, tmp_path):
        """Test generating quality report with output file."""
        assessment_results = client.comprehensive_assessment(mock_voxel_data, ["signal1"])
        output_file = tmp_path / "quality_report.txt"

        report = client.generate_quality_report(assessment_results, output_file=str(output_file))

        assert isinstance(report, str)
        assert output_file.exists()
        assert len(output_file.read_text()) > 0

    @pytest.mark.unit
    def test_client_creation_with_validation_enabled(self):
        """Test creating QualityAssessmentClient with validation enabled."""
        try:
            client = QualityAssessmentClient(enable_validation=True)
            assert client.enable_validation is True
            # validation_client may be None if validation module unavailable
        except TypeError:
            # If enable_validation parameter doesn't exist yet, skip
            pytest.skip("enable_validation parameter not available")

    @pytest.mark.unit
    def test_client_creation_with_validation_disabled(self):
        """Test creating QualityAssessmentClient with validation disabled."""
        try:
            client = QualityAssessmentClient(enable_validation=False)
            assert client.enable_validation is False
            assert client.validation_client is None
        except TypeError:
            pytest.skip("enable_validation parameter not available")

    @pytest.mark.unit
    def test_validate_quality_assessment_mpm(self, mock_voxel_data):
        """Test validate_quality_assessment with MPM validation type."""
        try:
            client = QualityAssessmentClient(enable_validation=True)
        except TypeError:
            pytest.skip("Validation features not available")

        if client.validation_client is None:
            pytest.skip("Validation client not available")

        mpm_reference = {
            "overall_quality_score": 0.88,
            "quality_scores": {
                "overall_quality_score": 0.88,
                "data_quality_score": 0.83,
            },
        }

        try:
            results = client.validate_quality_assessment(mock_voxel_data, mpm_reference, validation_type="mpm")

            assert isinstance(results, dict)
            assert "framework_metrics" in results
            assert "validation_type" in results
            assert results["validation_type"] == "mpm"
        except RuntimeError:
            pytest.skip("Validation not available")

    @pytest.mark.unit
    def test_validate_quality_assessment_raises_when_disabled(self, mock_voxel_data):
        """Test that validate_quality_assessment raises error when validation disabled."""
        try:
            client = QualityAssessmentClient(enable_validation=False)
        except TypeError:
            pytest.skip("Validation features not available")

        with pytest.raises(RuntimeError, match="Validation not available"):
            client.validate_quality_assessment(mock_voxel_data, {})

    @pytest.mark.unit
    def test_benchmark_quality_assessment(self, mock_voxel_data):
        """Test benchmark_quality_assessment method."""
        try:
            client = QualityAssessmentClient(enable_validation=True)
        except TypeError:
            pytest.skip("Validation features not available")

        if client.validation_client is None:
            pytest.skip("Validation client not available")

        result = client.benchmark_quality_assessment(mock_voxel_data, signals=["signal1"], iterations=2, warmup_iterations=1)

        if result is not None:
            assert hasattr(result, "execution_time")
            assert hasattr(result, "memory_usage")
            assert hasattr(result, "operation_name")

    @pytest.mark.unit
    def test_benchmark_quality_assessment_returns_none_when_disabled(self, mock_voxel_data):
        """Test benchmark_quality_assessment returns None when validation disabled."""
        try:
            client = QualityAssessmentClient(enable_validation=False)
        except TypeError:
            pytest.skip("Validation features not available")

        result = client.benchmark_quality_assessment(mock_voxel_data)

        assert result is None

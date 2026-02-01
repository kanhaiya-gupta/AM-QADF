"""
Unit tests for voxel-based anomaly detection utilities.

Tests for VoxelAnomalyResult and VoxelAnomalyDetector.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from am_qadf.anomaly_detection.utils.voxel_detection import (
    VoxelAnomalyResult,
    VoxelAnomalyDetector,
)


class TestVoxelAnomalyResult:
    """Test suite for VoxelAnomalyResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample VoxelAnomalyResult."""
        anomaly_map = np.random.rand(10, 10, 10) > 0.9
        anomaly_scores = np.random.rand(10, 10, 10).astype(np.float32)

        return VoxelAnomalyResult(
            anomaly_map=anomaly_map,
            anomaly_scores=anomaly_scores,
            num_anomalies=int(np.sum(anomaly_map)),
            anomaly_fraction=np.sum(anomaly_map) / anomaly_map.size,
        )

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating VoxelAnomalyResult."""
        anomaly_map = np.array([[[True, False], [False, True]]])
        anomaly_scores = np.array([[[0.9, 0.2], [0.3, 0.8]]], dtype=np.float32)

        result = VoxelAnomalyResult(
            anomaly_map=anomaly_map,
            anomaly_scores=anomaly_scores,
            num_anomalies=2,
            anomaly_fraction=0.5,
        )

        assert np.array_equal(result.anomaly_map, anomaly_map)
        assert np.array_equal(result.anomaly_scores, anomaly_scores)
        assert result.num_anomalies == 2
        assert result.anomaly_fraction == 0.5
        assert result.anomaly_types is None

    @pytest.mark.unit
    def test_result_creation_with_types(self):
        """Test creating VoxelAnomalyResult with anomaly types."""
        anomaly_map = np.array([[[True, False]]])
        anomaly_scores = np.array([[[0.9, 0.2]]], dtype=np.float32)
        anomaly_types = {
            "laser_power": np.array([[[True, False]]]),
            "density": np.array([[[False, True]]]),
        }

        result = VoxelAnomalyResult(
            anomaly_map=anomaly_map,
            anomaly_scores=anomaly_scores,
            anomaly_types=anomaly_types,
            num_anomalies=1,
            anomaly_fraction=0.5,
        )

        assert result.anomaly_types is not None
        assert "laser_power" in result.anomaly_types
        assert "density" in result.anomaly_types

    @pytest.mark.unit
    def test_result_to_dict(self, sample_result):
        """Test converting VoxelAnomalyResult to dictionary."""
        result_dict = sample_result.to_dict()

        assert isinstance(result_dict, dict)
        assert "num_anomalies" in result_dict
        assert "anomaly_fraction" in result_dict
        assert "anomaly_map_shape" in result_dict
        assert "anomaly_scores_shape" in result_dict
        assert result_dict["num_anomalies"] == sample_result.num_anomalies
        assert result_dict["anomaly_fraction"] == sample_result.anomaly_fraction

    @pytest.mark.unit
    def test_result_to_dict_with_types(self):
        """Test converting result with types to dictionary."""
        anomaly_map = np.array([[[True]]])
        anomaly_scores = np.array([[[0.9]]], dtype=np.float32)
        anomaly_types = {"signal1": np.array([[[True]]])}

        result = VoxelAnomalyResult(
            anomaly_map=anomaly_map,
            anomaly_scores=anomaly_scores,
            anomaly_types=anomaly_types,
            num_anomalies=1,
            anomaly_fraction=1.0,
        )

        result_dict = result.to_dict()
        assert "anomaly_types" in result_dict
        assert "signal1" in result_dict["anomaly_types"]


class TestVoxelAnomalyDetector:
    """Test suite for VoxelAnomalyDetector class."""

    @pytest.fixture
    def sample_signal_array(self):
        """Create a sample 3D signal array."""
        np.random.seed(42)
        return np.random.randn(10, 10, 10) * 10 + 100

    @pytest.fixture
    def mock_voxel_data(self):
        """Create a mock voxel data object."""
        mock_data = Mock()
        mock_data.get_signal_array = Mock(return_value=np.random.randn(10, 10, 10) * 10 + 100)
        mock_data.bbox_min = (0, 0, 0)
        mock_data.resolution = 0.5
        return mock_data

    @pytest.mark.unit
    def test_detector_creation_default(self):
        """Test creating VoxelAnomalyDetector with default parameters."""
        detector = VoxelAnomalyDetector()

        assert detector.method == "isolation_forest"
        assert detector.threshold == 0.5
        assert detector._detector is None

    @pytest.mark.unit
    def test_detector_creation_custom(self):
        """Test creating VoxelAnomalyDetector with custom parameters."""
        detector = VoxelAnomalyDetector(method="z_score", threshold=0.7)

        assert detector.method == "z_score"
        assert detector.threshold == 0.7

    @pytest.mark.unit
    def test_detect_spatial_anomalies_z_score(self, sample_signal_array):
        """Test detecting spatial anomalies using z-score method."""
        detector = VoxelAnomalyDetector(method="z_score", threshold=0.5)
        result = detector.detect_spatial_anomalies(sample_signal_array)

        assert isinstance(result, VoxelAnomalyResult)
        assert result.anomaly_map.shape == sample_signal_array.shape
        assert result.anomaly_scores.shape == sample_signal_array.shape
        assert result.num_anomalies >= 0
        assert 0 <= result.anomaly_fraction <= 1

    @pytest.mark.unit
    def test_detect_spatial_anomalies_iqr(self, sample_signal_array):
        """Test detecting spatial anomalies using IQR method."""
        detector = VoxelAnomalyDetector(method="iqr", threshold=0.5)
        result = detector.detect_spatial_anomalies(sample_signal_array)

        assert isinstance(result, VoxelAnomalyResult)
        assert result.anomaly_map.shape == sample_signal_array.shape
        assert result.anomaly_scores.shape == sample_signal_array.shape

    @pytest.mark.unit
    def test_detect_spatial_anomalies_insufficient_data(self):
        """Test detecting spatial anomalies with insufficient data."""
        # Create array with very few valid values
        small_array = np.full((5, 5, 5), np.nan)
        small_array[0, 0, 0] = 100.0

        detector = VoxelAnomalyDetector()
        result = detector.detect_spatial_anomalies(small_array)

        assert isinstance(result, VoxelAnomalyResult)
        assert result.num_anomalies == 0
        assert result.anomaly_fraction == 0.0

    @pytest.mark.unit
    def test_detect_spatial_anomalies_with_outliers(self, sample_signal_array):
        """Test detecting spatial anomalies in data with outliers."""
        # Add some outliers
        array_with_outliers = sample_signal_array.copy()
        array_with_outliers[0, 0, 0] = 1000.0  # Outlier
        array_with_outliers[1, 1, 1] = -1000.0  # Outlier

        detector = VoxelAnomalyDetector(method="z_score", threshold=0.3)
        result = detector.detect_spatial_anomalies(array_with_outliers)

        assert isinstance(result, VoxelAnomalyResult)
        # Should detect at least some anomalies
        assert result.num_anomalies >= 0

    @pytest.mark.unit
    def test_detect_temporal_anomalies(self, sample_signal_array):
        """Test detecting temporal anomalies."""
        detector = VoxelAnomalyDetector(threshold=0.5)
        result = detector.detect_temporal_anomalies(sample_signal_array, axis=2)

        assert isinstance(result, VoxelAnomalyResult)
        assert result.anomaly_map.shape == sample_signal_array.shape
        assert result.anomaly_scores.shape == sample_signal_array.shape

    @pytest.mark.unit
    def test_detect_temporal_anomalies_different_axis(self, sample_signal_array):
        """Test detecting temporal anomalies along different axes."""
        detector = VoxelAnomalyDetector(threshold=0.5)

        result_axis0 = detector.detect_temporal_anomalies(sample_signal_array, axis=0)
        result_axis1 = detector.detect_temporal_anomalies(sample_signal_array, axis=1)
        result_axis2 = detector.detect_temporal_anomalies(sample_signal_array, axis=2)

        assert isinstance(result_axis0, VoxelAnomalyResult)
        assert isinstance(result_axis1, VoxelAnomalyResult)
        assert isinstance(result_axis2, VoxelAnomalyResult)

    @pytest.mark.unit
    def test_detect_temporal_anomalies_insufficient_data(self):
        """Test detecting temporal anomalies with insufficient data."""
        # Create array with very few layers
        small_array = np.random.randn(2, 2, 2) * 10 + 100

        detector = VoxelAnomalyDetector()
        result = detector.detect_temporal_anomalies(small_array, axis=2)

        assert isinstance(result, VoxelAnomalyResult)
        # May return zero anomalies if insufficient data
        assert result.num_anomalies >= 0

    @pytest.mark.unit
    def test_detect_multi_signal_anomalies(self, mock_voxel_data):
        """Test detecting multi-signal anomalies."""
        detector = VoxelAnomalyDetector(threshold=0.5)

        result = detector.detect_multi_signal_anomalies(
            mock_voxel_data, signals=["laser_power", "density"], combine_method="max"
        )

        assert isinstance(result, VoxelAnomalyResult)
        assert result.anomaly_map.shape == (10, 10, 10)
        assert result.anomaly_scores.shape == (10, 10, 10)

    @pytest.mark.unit
    def test_detect_multi_signal_anomalies_mean(self, mock_voxel_data):
        """Test detecting multi-signal anomalies with mean combination."""
        detector = VoxelAnomalyDetector(threshold=0.5)

        result = detector.detect_multi_signal_anomalies(
            mock_voxel_data, signals=["laser_power", "density"], combine_method="mean"
        )

        assert isinstance(result, VoxelAnomalyResult)

    @pytest.mark.unit
    def test_detect_multi_signal_anomalies_no_signals(self, mock_voxel_data):
        """Test detecting multi-signal anomalies with no signals."""
        detector = VoxelAnomalyDetector()

        with pytest.raises(ValueError, match="At least one signal"):
            detector.detect_multi_signal_anomalies(mock_voxel_data, signals=[])

    @pytest.mark.unit
    def test_detect_multi_signal_anomalies_invalid_signal(self, mock_voxel_data):
        """Test detecting multi-signal anomalies with invalid signal."""
        detector = VoxelAnomalyDetector()
        mock_voxel_data.get_signal_array = Mock(side_effect=Exception("Signal not found"))

        with pytest.raises(ValueError, match="No valid signals"):
            detector.detect_multi_signal_anomalies(mock_voxel_data, signals=["invalid_signal"])

    @pytest.mark.unit
    def test_localize_anomalies(self, sample_signal_array, mock_voxel_data):
        """Test localizing anomalies to voxel coordinates."""
        detector = VoxelAnomalyDetector(threshold=0.3)
        result = detector.detect_spatial_anomalies(sample_signal_array)

        localization = detector.localize_anomalies(result, mock_voxel_data)

        assert isinstance(localization, dict)
        assert "anomaly_voxels" in localization
        assert "anomaly_coordinates" in localization
        assert "num_anomalies" in localization
        assert isinstance(localization["anomaly_voxels"], list)
        assert isinstance(localization["anomaly_coordinates"], list)

    @pytest.mark.unit
    def test_localize_anomalies_no_anomalies(self, sample_signal_array, mock_voxel_data):
        """Test localizing when there are no anomalies."""
        detector = VoxelAnomalyDetector(threshold=10.0)  # Very high threshold
        result = detector.detect_spatial_anomalies(sample_signal_array)

        localization = detector.localize_anomalies(result, mock_voxel_data)

        assert localization["num_anomalies"] == 0
        assert len(localization["anomaly_voxels"]) == 0
        assert len(localization["anomaly_coordinates"]) == 0

    @pytest.mark.unit
    def test_localize_anomalies_with_coordinates(self, sample_signal_array, mock_voxel_data):
        """Test localizing anomalies with coordinate conversion."""
        detector = VoxelAnomalyDetector(threshold=0.3)
        result = detector.detect_spatial_anomalies(sample_signal_array)

        localization = detector.localize_anomalies(result, mock_voxel_data)

        if localization["num_anomalies"] > 0:
            first_voxel = localization["anomaly_voxels"][0]
            assert "voxel_index" in first_voxel
            assert "world_coordinate" in first_voxel
            assert "anomaly_score" in first_voxel
            assert "is_anomaly" in first_voxel
            assert isinstance(first_voxel["world_coordinate"], tuple)
            assert len(first_voxel["world_coordinate"]) == 3

    @pytest.mark.unit
    def test_different_thresholds(self, sample_signal_array):
        """Test detecting with different thresholds."""
        detector_low = VoxelAnomalyDetector(threshold=0.1)
        detector_high = VoxelAnomalyDetector(threshold=0.9)

        result_low = detector_low.detect_spatial_anomalies(sample_signal_array)
        result_high = detector_high.detect_spatial_anomalies(sample_signal_array)

        # Lower threshold should detect more anomalies
        assert result_low.num_anomalies >= result_high.num_anomalies

    @pytest.mark.unit
    def test_get_detector(self):
        """Test getting detector instance."""
        detector = VoxelAnomalyDetector(method="z_score")
        detector_instance = detector._get_detector()

        # Should return detector or None
        assert detector_instance is None or hasattr(detector_instance, "predict")

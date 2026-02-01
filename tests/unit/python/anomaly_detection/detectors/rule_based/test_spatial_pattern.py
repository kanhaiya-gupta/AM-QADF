"""
Unit tests for Spatial Pattern detector.

Tests for SpatialPatternDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.rule_based.spatial_pattern import (
    SpatialPatternDetector,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


class TestSpatialPatternDetector:
    """Test suite for SpatialPatternDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create SpatialPatternDetector with default parameters."""
        return SpatialPatternDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create SpatialPatternDetector with custom parameters."""
        return SpatialPatternDetector(
            neighborhood_radius=1.0,
            use_density_analysis=True,
            use_gradient_analysis=False,
            use_clustering_analysis=True,
        )

    @pytest.fixture
    def normal_data(self):
        """Create normal spatial data for training."""
        np.random.seed(42)
        return np.random.randn(100, 3) * 10 + 100

    @pytest.fixture
    def data_with_spatial_anomalies(self, normal_data):
        """Create data with spatial anomalies."""
        data = normal_data.copy()
        # Add isolated point (spatial anomaly)
        data[0] = [200, 200, 200]  # Far from others
        # Add dense cluster (spatial anomaly)
        data[1:4] = [150, 150, 150]  # Dense cluster
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test SpatialPatternDetector initialization with default values."""
        detector = SpatialPatternDetector()

        assert detector.neighborhood_radius == 0.5
        assert detector.use_density_analysis is True
        assert detector.use_gradient_analysis is True
        assert detector.use_clustering_analysis is True
        assert detector.min_cluster_size == 3
        assert detector.threshold_percentile == 95.0
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.baseline_density_ is None
        assert detector.baseline_gradient_ is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test initialization with custom parameters."""
        detector = SpatialPatternDetector(neighborhood_radius=1.0, use_density_analysis=False, min_cluster_size=5)

        assert detector.neighborhood_radius == 1.0
        assert detector.use_density_analysis is False
        assert detector.min_cluster_size == 5

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = SpatialPatternDetector(config=config)

        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.coordinates_ is not None
        assert detector.coordinates_.shape[0] == normal_data.shape[0]
        assert detector.coordinates_.shape[1] == 3  # 3D coordinates
        assert detector is detector_default  # Method chaining

    @pytest.mark.unit
    def test_fit_baseline_density(self, detector_default, normal_data):
        """Test that baseline density is calculated."""
        detector = detector_default.fit(normal_data)

        if detector.use_density_analysis:
            assert detector.baseline_density_ is not None
            assert "mean" in detector.baseline_density_
            assert "std" in detector.baseline_density_
            assert detector.baseline_density_["mean"] >= 0

    @pytest.mark.unit
    def test_fit_baseline_gradient(self, detector_default, normal_data):
        """Test that baseline gradient is calculated."""
        detector = detector_default.fit(normal_data)

        if detector.use_gradient_analysis:
            assert detector.baseline_gradient_ is not None
            assert "mean" in detector.baseline_gradient_
            assert "std" in detector.baseline_gradient_

    @pytest.mark.unit
    def test_fit_without_density_analysis(self, normal_data):
        """Test fitting without density analysis."""
        detector = SpatialPatternDetector(use_density_analysis=False)
        detector.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.baseline_density_ is None

    @pytest.mark.unit
    def test_predict_before_fit(self, detector_default, normal_data):
        """Test that predict raises error if not fitted."""
        with pytest.raises(ValueError, match="must be fitted"):
            detector_default.predict(normal_data)

    @pytest.mark.unit
    def test_predict_normal_data(self, detector_default, normal_data):
        """Test prediction on normal data."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(normal_data)

        assert len(results) == len(normal_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)
        # Most points should not be spatial anomalies
        anomaly_count = sum(r.is_anomaly for r in results)
        assert anomaly_count < len(results) * 0.2

    @pytest.mark.unit
    def test_predict_with_spatial_anomalies(self, detector_default, normal_data, data_with_spatial_anomalies):
        """Test prediction on data with spatial anomalies."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_spatial_anomalies)

        assert len(results) == len(data_with_spatial_anomalies)
        # Isolated point should be detected
        assert results[0].is_anomaly is True or results[0].anomaly_score > results[10].anomaly_score

    @pytest.mark.unit
    def test_predict_density_analysis(self, detector_default, normal_data):
        """Test density-based anomaly detection."""
        detector = detector_default.fit(normal_data)

        # Create isolated point
        test_data = normal_data.copy()
        test_data[0] = [1000, 1000, 1000]  # Far isolated point

        results = detector.predict(test_data)

        # Isolated point should have low density
        assert results[0].anomaly_score > results[5].anomaly_score

    @pytest.mark.unit
    def test_predict_gradient_analysis(self, detector_default, normal_data):
        """Test gradient-based anomaly detection."""
        detector = detector_default.fit(normal_data)

        # Create point with high gradient
        test_data = normal_data.copy()
        test_data[0] = [200, 200, 200]  # High gradient from neighbors

        results = detector.predict(test_data)

        # High gradient point should be detected
        assert results[0].anomaly_score > results[5].anomaly_score

    @pytest.mark.unit
    def test_predict_clustering_analysis(self, detector_default, normal_data):
        """Test clustering-based anomaly detection."""
        detector = SpatialPatternDetector(use_clustering_analysis=True)
        detector.fit(normal_data)

        # Create isolated point
        test_data = normal_data.copy()
        test_data[0] = [1000, 1000, 1000]

        results = detector.predict(test_data)

        # Isolated point should be detected
        assert len(results) == len(test_data)

    @pytest.mark.unit
    def test_predict_scores(self, detector_default, normal_data, data_with_spatial_anomalies):
        """Test predict_scores method."""
        detector = detector_default.fit(normal_data)
        scores = detector.predict_scores(data_with_spatial_anomalies)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data_with_spatial_anomalies)
        assert all(s >= 0 for s in scores)
        assert all(s <= 1 for s in scores)  # Normalized

    @pytest.mark.unit
    def test_different_neighborhood_radius(self, normal_data):
        """Test with different neighborhood radius values."""
        detector_small = SpatialPatternDetector(neighborhood_radius=0.1).fit(normal_data)
        detector_large = SpatialPatternDetector(neighborhood_radius=2.0).fit(normal_data)

        test_data = normal_data.copy()
        test_data[0] = [200, 200, 200]

        results_small = detector_small.predict(test_data)
        results_large = detector_large.predict(test_data)

        # Both should work
        assert len(results_small) == len(results_large)

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.random.randn(10, 3) * 10 + 100

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_edge_case_single_point(self, detector_default):
        """Test with single data point."""
        single_point = np.array([[100, 100, 100]])

        detector = detector_default.fit(single_point)
        results = detector.predict(single_point)

        assert len(results) == 1
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

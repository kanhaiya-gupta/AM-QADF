"""
Unit tests for Weighted Ensemble detector.

Tests for WeightedEnsembleDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from am_qadf.anomaly_detection.detectors.ensemble.weighted_ensemble import (
    WeightedEnsembleDetector,
)
from am_qadf.anomaly_detection.core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)


class MockDetector(BaseAnomalyDetector):
    """Mock detector for testing ensemble."""

    def __init__(self, name="MockDetector", score=0.5):
        super().__init__()
        self.name = name
        self.score = score
        self.fit_called = False

    def fit(self, data, labels=None):
        self.fit_called = True
        self.is_fitted = True
        return self

    def predict(self, data):
        # Return mock results
        n_samples = len(data) if hasattr(data, "__len__") else 1
        results = []
        for i in range(n_samples):
            results.append(
                AnomalyDetectionResult(
                    voxel_index=(0, 0, 0),
                    voxel_coordinates=(0.0, 0.0, 0.0),
                    is_anomaly=self.score >= 0.5,
                    anomaly_score=self.score,
                    confidence=0.9,
                    detector_name=self.name,
                )
            )
        return results


class TestWeightedEnsembleDetector:
    """Test suite for WeightedEnsembleDetector class."""

    @pytest.fixture
    def mock_detectors(self):
        """Create mock detectors for testing."""
        return [
            MockDetector(name="Detector1", score=0.3),
            MockDetector(name="Detector2", score=0.5),
            MockDetector(name="Detector3", score=0.7),
        ]

    @pytest.fixture
    def normal_data(self):
        """Create normal data for training."""
        np.random.seed(42)
        return np.random.randn(50, 3) * 10 + 100

    @pytest.mark.unit
    def test_detector_initialization_default(self, mock_detectors):
        """Test WeightedEnsembleDetector initialization with default values."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)

        assert len(ensemble.detectors) == 3
        assert ensemble.weights is not None
        # Should have uniform weights
        assert len(ensemble.weights) == 3
        assert np.allclose(ensemble.weights, [1 / 3, 1 / 3, 1 / 3])
        assert ensemble.config is not None

    @pytest.mark.unit
    def test_detector_initialization_uniform_weights(self, mock_detectors):
        """Test initialization with uniform weights (default)."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)

        # Weights should sum to 1
        assert abs(np.sum(ensemble.weights) - 1.0) < 1e-10
        # All weights should be equal
        assert np.allclose(ensemble.weights, ensemble.weights[0])

    @pytest.mark.unit
    def test_detector_initialization_custom_weights(self, mock_detectors):
        """Test initialization with custom weights."""
        custom_weights = [0.5, 0.3, 0.2]
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors, weights=custom_weights)

        # Weights should be normalized
        assert abs(np.sum(ensemble.weights) - 1.0) < 1e-10
        # Should match normalized custom weights
        expected = np.array(custom_weights) / np.sum(custom_weights)
        assert np.allclose(ensemble.weights, expected)

    @pytest.mark.unit
    def test_detector_initialization_custom_weights_no_normalize(self, mock_detectors):
        """Test initialization with custom weights without normalization."""
        custom_weights = [0.5, 0.3, 0.2]
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors, weights=custom_weights, normalize_weights=False)

        # Weights should not be normalized
        assert np.allclose(ensemble.weights, custom_weights)

    @pytest.mark.unit
    def test_detector_initialization_with_config(self, mock_detectors):
        """Test initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.6)
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors, config=config)

        assert ensemble.config.threshold == 0.6

    @pytest.mark.unit
    def test_fit(self, mock_detectors, normal_data):
        """Test fitting the ensemble."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)
        ensemble.fit(normal_data)

        assert ensemble.is_fitted is True
        # All detectors should be fitted
        assert all(d.fit_called for d in mock_detectors)
        assert all(d.is_fitted for d in mock_detectors)

    @pytest.mark.unit
    def test_fit_with_labels(self, mock_detectors, normal_data):
        """Test fitting with labels."""
        labels = np.random.randint(0, 2, len(normal_data))
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)

        ensemble.fit(normal_data, labels)

        assert ensemble.is_fitted is True

    @pytest.mark.unit
    def test_set_weights_from_performance(self, mock_detectors):
        """Test setting weights from performance scores."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)
        performance_scores = [0.8, 0.6, 0.9]  # Detector3 is best

        ensemble.set_weights_from_performance(performance_scores)

        # Weights should be proportional to performance
        assert abs(np.sum(ensemble.weights) - 1.0) < 1e-10
        # Best detector should have highest weight
        assert ensemble.weights[2] > ensemble.weights[1]
        assert ensemble.weights[2] > ensemble.weights[0]

    @pytest.mark.unit
    def test_set_weights_from_performance_wrong_length(self, mock_detectors):
        """Test that wrong number of performance scores raises error."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)
        performance_scores = [0.8, 0.6]  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            ensemble.set_weights_from_performance(performance_scores)

    @pytest.mark.unit
    def test_set_weights_from_performance_negative(self, mock_detectors):
        """Test that negative performance scores are handled."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)
        performance_scores = [0.8, -0.1, 0.9]  # One negative

        ensemble.set_weights_from_performance(performance_scores)

        # Negative scores should be clamped to 0
        assert all(w >= 0 for w in ensemble.weights)
        assert abs(np.sum(ensemble.weights) - 1.0) < 1e-10

    @pytest.mark.unit
    def test_set_weights_from_performance_all_zero(self, mock_detectors):
        """Test that all zero performance scores result in uniform weights."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)
        performance_scores = [0.0, 0.0, 0.0]

        ensemble.set_weights_from_performance(performance_scores)

        # Should fall back to uniform weights
        assert np.allclose(ensemble.weights, [1 / 3, 1 / 3, 1 / 3])

    @pytest.mark.unit
    def test_predict_before_fit(self, mock_detectors, normal_data):
        """Test that predict raises error if not fitted."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)

        with pytest.raises(ValueError, match="must be fitted"):
            ensemble.predict(normal_data)

    @pytest.mark.unit
    def test_predict_weighted_average(self, mock_detectors, normal_data):
        """Test that scores are weighted averaged correctly."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # Scores: 0.3, 0.5, 0.7 with uniform weights (1/3 each)
        # Weighted average = (0.3 + 0.5 + 0.7) / 3 = 0.5
        expected_score = (0.3 + 0.5 + 0.7) / 3
        assert abs(results[0].anomaly_score - expected_score) < 1e-10

    @pytest.mark.unit
    def test_predict_custom_weights(self, mock_detectors, normal_data):
        """Test prediction with custom weights."""
        custom_weights = [0.5, 0.3, 0.2]
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors, weights=custom_weights)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # Weighted average: 0.3*0.5 + 0.5*0.3 + 0.7*0.2 = 0.15 + 0.15 + 0.14 = 0.44
        # But weights are normalized, so: [0.5, 0.3, 0.2] -> [0.5, 0.3, 0.2] (sum=1.0)
        expected_score = 0.3 * 0.5 + 0.5 * 0.3 + 0.7 * 0.2
        assert abs(results[0].anomaly_score - expected_score) < 1e-10

    @pytest.mark.unit
    def test_predict_anomaly_detection(self, mock_detectors, normal_data):
        """Test that anomaly detection uses threshold."""
        config = AnomalyDetectionConfig(threshold=0.4)
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors, config=config)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # Weighted score = 0.5, threshold = 0.4
        assert results[0].anomaly_score == 0.5
        assert results[0].is_anomaly is True  # 0.5 >= 0.4

    @pytest.mark.unit
    def test_predict_confidence_calculation(self, mock_detectors, normal_data):
        """Test that confidence is calculated from score agreement."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # Confidence should be based on score std
        # Lower std (more agreement) = higher confidence
        assert 0 <= results[0].confidence <= 1

    @pytest.mark.unit
    def test_predict_metadata(self, mock_detectors, normal_data):
        """Test that metadata includes score and weight information."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        assert "individual_scores" in results[0].metadata
        assert "weights" in results[0].metadata
        assert len(results[0].metadata["individual_scores"]) == 3
        assert len(results[0].metadata["weights"]) == 3
        assert results[0].metadata["n_detectors"] == 3
        assert results[0].metadata["individual_scores"] == [0.3, 0.5, 0.7]

    @pytest.mark.unit
    def test_predict_multiple_samples(self, mock_detectors, normal_data):
        """Test prediction on multiple samples."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:5])

        assert len(results) == 5
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_predict_after_set_weights(self, mock_detectors, normal_data):
        """Test prediction after setting weights from performance."""
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors)
        ensemble.fit(normal_data)

        # Set weights based on performance
        performance_scores = [0.8, 0.6, 0.9]
        ensemble.set_weights_from_performance(performance_scores)

        results = ensemble.predict(normal_data[:1])

        # Should use updated weights
        assert len(results) == 1
        # Detector3 (score=0.7) should have higher weight now
        assert results[0].anomaly_score > 0.5  # Should be closer to 0.7

    @pytest.mark.unit
    def test_edge_case_single_detector(self, normal_data):
        """Test with single detector."""
        detector = MockDetector(name="SingleDetector", score=0.6)
        ensemble = WeightedEnsembleDetector(detectors=[detector])
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        assert len(results) == 1
        # Single detector's score is the ensemble score
        assert results[0].anomaly_score == 0.6
        assert results[0].is_anomaly is True  # 0.6 >= 0.5 (default threshold)

    @pytest.mark.unit
    def test_edge_case_high_variance_scores(self, normal_data):
        """Test with detectors that have very different scores."""
        detectors = [
            MockDetector(name="Detector1", score=0.1),
            MockDetector(name="Detector2", score=0.9),
        ]

        ensemble = WeightedEnsembleDetector(detectors=detectors)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # Weighted average: (0.1 + 0.9) / 2 = 0.5
        assert abs(results[0].anomaly_score - 0.5) < 1e-10
        # High variance should result in lower confidence
        assert results[0].confidence < 1.0

    @pytest.mark.unit
    def test_edge_case_low_variance_scores(self, normal_data):
        """Test with detectors that have similar scores."""
        detectors = [
            MockDetector(name="Detector1", score=0.49),
            MockDetector(name="Detector2", score=0.51),
        ]

        ensemble = WeightedEnsembleDetector(detectors=detectors)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # Low variance should result in higher confidence
        assert results[0].confidence > 0.5

    @pytest.mark.unit
    def test_edge_case_zero_weights(self, mock_detectors, normal_data):
        """Test that zero weights are handled."""
        # This might cause issues, but should be handled gracefully
        custom_weights = [1.0, 0.0, 0.0]
        ensemble = WeightedEnsembleDetector(detectors=mock_detectors, weights=custom_weights, normalize_weights=True)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # Should only use first detector's score
        assert abs(results[0].anomaly_score - 0.3) < 1e-10

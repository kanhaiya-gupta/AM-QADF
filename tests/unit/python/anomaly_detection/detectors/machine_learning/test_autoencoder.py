"""
Unit tests for Autoencoder detector.

Tests for AutoencoderDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.machine_learning.autoencoder import (
    AutoencoderDetector,
    TENSORFLOW_AVAILABLE,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow/Keras not available")
class TestAutoencoderDetector:
    """Test suite for AutoencoderDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create AutoencoderDetector with default parameters."""
        return AutoencoderDetector(epochs=5)  # Reduced epochs for faster tests

    @pytest.fixture
    def detector_custom(self):
        """Create AutoencoderDetector with custom parameters."""
        return AutoencoderDetector(
            encoding_dim=8,
            hidden_layers=[32, 16],
            activation="tanh",
            epochs=5,
            batch_size=16,
        )

    @pytest.fixture
    def normal_data(self):
        """Create normal data for training."""
        np.random.seed(42)
        return np.random.randn(100, 10) * 10 + 100

    @pytest.fixture
    def data_with_outliers(self, normal_data):
        """Create data with clear outliers."""
        data = normal_data.copy()
        # Add clear outliers
        data[0] = np.random.randn(10) * 50 + 200  # High outlier
        data[1] = np.random.randn(10) * 50 - 50  # Low outlier
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test AutoencoderDetector initialization with default values."""
        detector = AutoencoderDetector()

        assert detector.encoding_dim is None
        assert detector.hidden_layers == [64, 32, 16]
        assert detector.activation == "relu"
        assert detector.optimizer == "adam"
        assert detector.epochs == 50
        assert detector.batch_size == 32
        assert detector.threshold_percentile == 95.0
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.model is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test AutoencoderDetector initialization with custom parameters."""
        detector = AutoencoderDetector(
            encoding_dim=8,
            hidden_layers=[32, 16],
            activation="tanh",
            optimizer="sgd",
            epochs=20,
            batch_size=16,
        )

        assert detector.encoding_dim == 8
        assert detector.hidden_layers == [32, 16]
        assert detector.activation == "tanh"
        assert detector.optimizer == "sgd"
        assert detector.epochs == 20
        assert detector.batch_size == 16

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test AutoencoderDetector initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = AutoencoderDetector(epochs=5, config=config)

        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.model is not None
        assert detector.input_dim == normal_data.shape[1]
        assert detector.threshold_ is not None
        assert detector is detector_default  # Method chaining

    @pytest.mark.unit
    def test_fit_with_dataframe(self, detector_default):
        """Test fitting with pandas DataFrame."""
        df = pd.DataFrame({f"feature{i}": np.random.randn(50) * 10 + 100 for i in range(5)})

        detector = detector_default.fit(df)

        assert detector.is_fitted is True
        assert detector.input_dim == 5

    @pytest.mark.unit
    def test_fit_auto_encoding_dim(self, normal_data):
        """Test that encoding_dim is auto-calculated if None."""
        detector = AutoencoderDetector(encoding_dim=None, epochs=5)
        detector.fit(normal_data)

        # Should be max(2, input_dim // 4) = max(2, 10//4) = max(2, 2) = 2
        assert detector.model is not None

    @pytest.mark.unit
    def test_fit_threshold_calculation(self, detector_default, normal_data):
        """Test that threshold is calculated from reconstruction errors."""
        detector = detector_default.fit(normal_data)

        assert detector.threshold_ is not None
        assert detector.threshold_ > 0
        # Threshold should be set in config
        assert detector.config.threshold == detector.threshold_

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
        # Most points should not be anomalies
        anomaly_count = sum(r.is_anomaly for r in results)
        assert anomaly_count < len(results) * 0.2  # Less than 20% anomalies

    @pytest.mark.unit
    def test_predict_with_outliers(self, detector_default, normal_data, data_with_outliers):
        """Test prediction on data with clear outliers."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_outliers)

        assert len(results) == len(data_with_outliers)
        # Outliers should have higher reconstruction errors
        assert results[0].anomaly_score > results[5].anomaly_score
        assert results[1].anomaly_score > results[5].anomaly_score

    @pytest.mark.unit
    def test_predict_dimension_mismatch(self, detector_default, normal_data):
        """Test that dimension mismatch raises error."""
        detector = detector_default.fit(normal_data)

        # Create data with wrong dimension
        wrong_dim_data = np.random.randn(10, 5)  # Should be 10 features

        with pytest.raises(ValueError, match="dimension mismatch"):
            detector.predict(wrong_dim_data)

    @pytest.mark.unit
    def test_predict_reconstruction_error(self, detector_default, normal_data):
        """Test that reconstruction error is used as anomaly score."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(normal_data[:5])

        # Scores should be reconstruction errors (non-negative)
        assert all(r.anomaly_score >= 0 for r in results)

    @pytest.mark.unit
    def test_predict_scores(self, detector_default, normal_data, data_with_outliers):
        """Test predict_scores method."""
        detector = detector_default.fit(normal_data)
        scores = detector.predict_scores(data_with_outliers)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data_with_outliers)
        assert all(s >= 0 for s in scores)
        # Outliers should have higher scores
        assert scores[0] > scores[5]
        assert scores[1] > scores[5]

    @pytest.mark.unit
    def test_different_hidden_layers(self, normal_data):
        """Test with different hidden layer configurations."""
        detector = AutoencoderDetector(hidden_layers=[128, 64, 32, 16], epochs=5)
        detector.fit(normal_data)
        results = detector.predict(normal_data[:10])

        assert len(results) == 10
        assert detector.is_fitted is True

    @pytest.mark.unit
    def test_different_activations(self, normal_data):
        """Test with different activation functions."""
        for activation in ["relu", "tanh", "sigmoid"]:
            detector = AutoencoderDetector(activation=activation, epochs=5)
            detector.fit(normal_data)
            assert detector.is_fitted is True

    @pytest.mark.unit
    def test_different_optimizers(self, normal_data):
        """Test with different optimizers."""
        for optimizer in ["adam", "sgd", "rmsprop"]:
            detector = AutoencoderDetector(optimizer=optimizer, epochs=5)
            detector.fit(normal_data)
            assert detector.is_fitted is True

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.random.randn(20, 5) * 10 + 100

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_edge_case_single_feature(self, detector_default):
        """Test with single feature."""
        single_feature = np.random.randn(50, 1) * 10 + 100

        detector = detector_default.fit(single_feature)
        results = detector.predict(single_feature)

        assert len(results) == len(single_feature)
        assert detector.input_dim == 1

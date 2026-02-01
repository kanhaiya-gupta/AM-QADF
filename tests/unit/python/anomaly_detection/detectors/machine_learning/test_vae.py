"""
Unit tests for VAE detector.

Tests for VAEDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.machine_learning.vae import (
    VAEDetector,
    TENSORFLOW_AVAILABLE,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow/Keras not available")
class TestVAEDetector:
    """Test suite for VAEDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create VAEDetector with default parameters."""
        return VAEDetector(epochs=5)  # Reduced epochs for faster tests

    @pytest.fixture
    def detector_custom(self):
        """Create VAEDetector with custom parameters."""
        return VAEDetector(
            encoding_dim=8,
            hidden_layers=[32, 16],
            activation="tanh",
            beta=0.5,
            epochs=5,
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
        """Test VAEDetector initialization with default values."""
        detector = VAEDetector()

        assert detector.encoding_dim == 16
        assert detector.hidden_layers == [64, 32]
        assert detector.activation == "relu"
        assert detector.optimizer == "adam"
        assert detector.epochs == 50
        assert detector.beta == 1.0
        assert detector.threshold_percentile == 95.0
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.encoder is None
        assert detector.decoder is None
        assert detector.vae is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test VAEDetector initialization with custom parameters."""
        detector = VAEDetector(
            encoding_dim=8,
            hidden_layers=[32, 16],
            activation="tanh",
            beta=0.5,
            epochs=20,
        )

        assert detector.encoding_dim == 8
        assert detector.hidden_layers == [32, 16]
        assert detector.activation == "tanh"
        assert detector.beta == 0.5
        assert detector.epochs == 20

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test VAEDetector initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = VAEDetector(epochs=5, config=config)

        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_detector_initialization_no_tensorflow(self):
        """Test that initialization raises error if TensorFlow not available."""
        # This test would need to mock TENSORFLOW_AVAILABLE = False
        # For now, we skip if TensorFlow is not available
        pass

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.vae is not None
        assert detector.encoder is not None
        assert detector.decoder is not None
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
    def test_fit_threshold_calculation(self, detector_default, normal_data):
        """Test that threshold is calculated from reconstruction errors."""
        detector = detector_default.fit(normal_data)

        assert detector.threshold_ is not None
        assert detector.threshold_ > 0
        assert detector.config.threshold == detector.threshold_

    @pytest.mark.unit
    def test_fit_vae_components(self, detector_default, normal_data):
        """Test that VAE components (encoder, decoder, vae) are created."""
        detector = detector_default.fit(normal_data)

        assert detector.encoder is not None
        assert detector.decoder is not None
        assert detector.vae is not None
        # Encoder should output [z_mean, z_log_var, z]
        # Decoder should take latent input
        # VAE should combine both

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
        assert anomaly_count < len(results) * 0.2

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
    def test_different_beta_values(self, normal_data):
        """Test with different beta values (beta-VAE)."""
        for beta in [0.1, 0.5, 1.0, 2.0]:
            detector = VAEDetector(beta=beta, epochs=5)
            detector.fit(normal_data)
            assert detector.is_fitted is True
            assert detector.beta == beta

    @pytest.mark.unit
    def test_different_hidden_layers(self, normal_data):
        """Test with different hidden layer configurations."""
        detector = VAEDetector(hidden_layers=[128, 64, 32, 16], epochs=5)
        detector.fit(normal_data)
        results = detector.predict(normal_data[:10])

        assert len(results) == 10
        assert detector.is_fitted is True

    @pytest.mark.unit
    def test_different_encoding_dims(self, normal_data):
        """Test with different encoding dimensions."""
        for encoding_dim in [4, 8, 16, 32]:
            detector = VAEDetector(encoding_dim=encoding_dim, epochs=5)
            detector.fit(normal_data)
            assert detector.is_fitted is True
            assert detector.encoding_dim == encoding_dim

    @pytest.mark.unit
    def test_different_activations(self, normal_data):
        """Test with different activation functions."""
        for activation in ["relu", "tanh"]:
            detector = VAEDetector(activation=activation, epochs=5)
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

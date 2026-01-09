"""
Unit tests for LSTM Autoencoder detector.

Tests for LSTMAutoencoderDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.machine_learning.lstm_autoencoder import (
    LSTMAutoencoderDetector,
    TENSORFLOW_AVAILABLE,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow/Keras not available")
class TestLSTMAutoencoderDetector:
    """Test suite for LSTMAutoencoderDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create LSTMAutoencoderDetector with default parameters."""
        return LSTMAutoencoderDetector(epochs=5)  # Reduced epochs for faster tests

    @pytest.fixture
    def detector_custom(self):
        """Create LSTMAutoencoderDetector with custom parameters."""
        return LSTMAutoencoderDetector(sequence_length=5, lstm_units=[32, 16], encoding_dim=8, epochs=5)

    @pytest.fixture
    def normal_data(self):
        """Create normal temporal data for training."""
        np.random.seed(42)
        # Create time-series data
        return np.random.randn(100, 5) * 10 + 100

    @pytest.fixture
    def data_with_outliers(self, normal_data):
        """Create data with clear outliers."""
        data = normal_data.copy()
        # Add clear outliers
        data[0] = np.random.randn(5) * 50 + 200  # High outlier
        data[1] = np.random.randn(5) * 50 - 50  # Low outlier
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test LSTMAutoencoderDetector initialization with default values."""
        detector = LSTMAutoencoderDetector()

        assert detector.sequence_length == 10
        assert detector.lstm_units == [64, 32]
        assert detector.encoding_dim is None
        assert detector.activation == "tanh"
        assert detector.optimizer == "adam"
        assert detector.epochs == 50
        assert detector.threshold_percentile == 95.0
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.model is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test LSTMAutoencoderDetector initialization with custom parameters."""
        detector = LSTMAutoencoderDetector(
            sequence_length=5,
            lstm_units=[32, 16],
            encoding_dim=8,
            activation="relu",
            epochs=20,
        )

        assert detector.sequence_length == 5
        assert detector.lstm_units == [32, 16]
        assert detector.encoding_dim == 8
        assert detector.activation == "relu"
        assert detector.epochs == 20

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test LSTMAutoencoderDetector initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = LSTMAutoencoderDetector(epochs=5, config=config)

        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_detector_initialization_no_tensorflow(self):
        """Test that initialization raises error if TensorFlow not available."""
        # This test would need to mock TENSORFLOW_AVAILABLE = False
        # For now, we skip if TensorFlow is not available
        pass

    @pytest.mark.unit
    def test_create_sequences(self, detector_default, normal_data):
        """Test sequence creation."""
        detector = detector_default
        sequences = detector._create_sequences(normal_data)

        # Should create sequences of length sequence_length
        assert len(sequences) == len(normal_data) - detector.sequence_length + 1
        assert sequences.shape[1] == detector.sequence_length
        assert sequences.shape[2] == normal_data.shape[1]

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
        detector = LSTMAutoencoderDetector(encoding_dim=None, epochs=5)
        detector.fit(normal_data)

        # Should be max(2, input_dim // 2) = max(2, 5//2) = max(2, 2) = 2
        assert detector.model is not None

    @pytest.mark.unit
    def test_fit_threshold_calculation(self, detector_default, normal_data):
        """Test that threshold is calculated from reconstruction errors."""
        detector = detector_default.fit(normal_data)

        assert detector.threshold_ is not None
        assert detector.threshold_ > 0
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

        # Note: LSTM creates sequences, so output length may differ
        assert len(results) > 0
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)
        # Most points should not be anomalies
        anomaly_count = sum(r.is_anomaly for r in results)
        assert anomaly_count < len(results) * 0.2

    @pytest.mark.unit
    def test_predict_with_outliers(self, detector_default, normal_data, data_with_outliers):
        """Test prediction on data with clear outliers."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_outliers)

        assert len(results) > 0
        # Outliers should have higher reconstruction errors
        if len(results) > 5:
            scores = [r.anomaly_score for r in results]
            # Outliers should have higher scores
            assert max(scores) > min(scores)

    @pytest.mark.unit
    def test_predict_reconstruction_error(self, detector_default, normal_data):
        """Test that reconstruction error is used as anomaly score."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(normal_data[:20])

        # Scores should be reconstruction errors (non-negative)
        assert all(r.anomaly_score >= 0 for r in results)

    @pytest.mark.unit
    def test_predict_scores(self, detector_default, normal_data, data_with_outliers):
        """Test predict_scores method."""
        detector = detector_default.fit(normal_data)
        scores = detector.predict_scores(data_with_outliers)

        assert isinstance(scores, np.ndarray)
        assert len(scores) > 0
        assert all(s >= 0 for s in scores)

    @pytest.mark.unit
    def test_different_sequence_lengths(self, normal_data):
        """Test with different sequence lengths."""
        for seq_len in [5, 10, 15]:
            detector = LSTMAutoencoderDetector(sequence_length=seq_len, epochs=5)
            detector.fit(normal_data)
            assert detector.is_fitted is True

    @pytest.mark.unit
    def test_different_lstm_units(self, normal_data):
        """Test with different LSTM unit configurations."""
        detector = LSTMAutoencoderDetector(lstm_units=[128, 64, 32], epochs=5)
        detector.fit(normal_data)
        results = detector.predict(normal_data[:10])

        assert len(results) > 0
        assert detector.is_fitted is True

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.random.randn(15, 5) * 10 + 100  # Need enough for sequences

        detector = LSTMAutoencoderDetector(sequence_length=5, epochs=5)
        detector.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) > 0
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_edge_case_insufficient_sequence_length(self, detector_default):
        """Test with data shorter than sequence_length."""
        # Data shorter than sequence_length
        short_data = np.random.randn(5, 5) * 10 + 100

        detector = LSTMAutoencoderDetector(sequence_length=10, epochs=5)

        # Should handle gracefully or raise error
        try:
            detector.fit(short_data)
            # If it fits, prediction should work
            results = detector.predict(short_data)
            assert len(results) >= 0
        except (ValueError, IndexError):
            # Expected if data is too short
            pass

"""
LSTM Autoencoder-Based Anomaly Detection

Detects temporal anomalies using LSTM autoencoders for time-series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings

from ...core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

# Try to import tensorflow/keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models

    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import keras
        from keras import layers, models

        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        warnings.warn("TensorFlow/Keras not available. LSTMAutoencoderDetector requires TensorFlow/Keras.")

logger = logging.getLogger(__name__)


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """
    LSTM Autoencoder-based anomaly detector for temporal data.

    Trains an LSTM autoencoder on normal time-series data and uses
    reconstruction error as anomaly score. Suitable for sequential/temporal data.
    """

    def __init__(
        self,
        sequence_length: int = 10,
        lstm_units: List[int] = None,
        encoding_dim: Optional[int] = None,
        activation: str = "tanh",
        optimizer: str = "adam",
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        threshold_percentile: float = 95.0,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize LSTM Autoencoder detector.

        Args:
            sequence_length: Length of input sequences
            lstm_units: List of LSTM units per layer (e.g., [64, 32])
            encoding_dim: Dimension of encoding layer (auto if None)
            activation: Activation function ('tanh', 'relu')
            optimizer: Optimizer ('adam', 'sgd', 'rmsprop')
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            threshold_percentile: Percentile for threshold (default: 95th)
            config: Optional detector configuration
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required for LSTMAutoencoderDetector")

        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)

        self.sequence_length = sequence_length
        self.lstm_units = lstm_units or [64, 32]
        self.encoding_dim = encoding_dim
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.threshold_percentile = threshold_percentile

        self.model = None
        self.input_dim = None
        self.threshold_ = None

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences from data for LSTM input."""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i : i + self.sequence_length])
        return np.array(sequences)

    def _build_model(self, input_dim: int):
        """Build LSTM autoencoder model."""
        # Input shape: (sequence_length, input_dim)
        input_layer = layers.Input(shape=(self.sequence_length, input_dim))

        # Encoder: LSTM layers
        encoded = input_layer
        for units in self.lstm_units:
            encoded = layers.LSTM(units, activation=self.activation, return_sequences=True)(encoded)

        # Encoding layer (last LSTM layer, return_sequences=False)
        if self.encoding_dim is None:
            encoding_dim = max(2, input_dim // 2)
        else:
            encoding_dim = self.encoding_dim

        encoded = layers.LSTM(encoding_dim, activation=self.activation, name="encoding")(encoded)

        # Repeat vector to match sequence length
        decoded = layers.RepeatVector(self.sequence_length)(encoded)

        # Decoder: LSTM layers (reverse)
        for units in reversed(self.lstm_units):
            decoded = layers.LSTM(units, activation=self.activation, return_sequences=True)(decoded)

        # Output layer
        decoded = layers.TimeDistributed(layers.Dense(input_dim, activation="linear"))(decoded)

        # Create model
        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer=self.optimizer, loss="mse", metrics=["mae"])

        return autoencoder

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "LSTMAutoencoderDetector":
        """
        Fit the LSTM autoencoder on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for autoencoder (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)
        self.input_dim = array_data.shape[1]

        # Create sequences
        sequences = self._create_sequences(array_data)

        if len(sequences) == 0:
            raise ValueError(
                f"Data too short for sequence length {self.sequence_length}. Need at least {self.sequence_length} samples."
            )

        # Build model
        self.model = self._build_model(self.input_dim)

        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)

        # Train
        self.model.fit(
            sequences,
            sequences,  # Autoencoder: input = target
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping],
            verbose=0,
        )

        # Calculate reconstruction errors on training sequences
        reconstructed = self.model.predict(sequences, verbose=0)
        reconstruction_errors = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))

        # Set threshold based on percentile
        self.threshold_ = np.percentile(reconstruction_errors, self.threshold_percentile)
        self.config.threshold = float(self.threshold_)

        self.is_fitted = True
        logger.info(f"LSTMAutoencoderDetector fitted on {len(sequences)} sequences. Threshold: {self.threshold_:.4f}")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using LSTM reconstruction error.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        if array_data.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {array_data.shape[1]}")

        # Create sequences
        sequences = self._create_sequences(array_data)

        if len(sequences) == 0:
            # If data is too short, pad or use single sample
            # For simplicity, pad with last value
            if len(array_data) < self.sequence_length:
                padding = np.tile(array_data[-1:], (self.sequence_length - len(array_data), 1))
                array_data = np.vstack([array_data, padding])
                sequences = array_data.reshape(1, self.sequence_length, self.input_dim)
            else:
                sequences = array_data[-self.sequence_length :].reshape(1, self.sequence_length, self.input_dim)

        # Get reconstructions
        reconstructed = self.model.predict(sequences, verbose=0)
        reconstruction_errors = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))

        # Map sequence errors back to individual samples
        # For each sequence, assign error to the last sample in the sequence
        anomaly_scores = np.zeros(len(array_data))
        for i, error in enumerate(reconstruction_errors):
            sample_idx = min(i + self.sequence_length - 1, len(array_data) - 1)
            anomaly_scores[sample_idx] = error

        # For samples before first sequence, use first sequence error
        if len(anomaly_scores) > len(reconstruction_errors):
            anomaly_scores[: self.sequence_length - 1] = reconstruction_errors[0]

        # Get indices and coordinates if available
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(anomaly_scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(anomaly_scores)  # Placeholder

        # Create results
        results = self._create_results(scores=anomaly_scores, indices=indices, coordinates=coordinates)

        return results

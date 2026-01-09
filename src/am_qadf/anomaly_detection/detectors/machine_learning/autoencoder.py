"""
Autoencoder-Based Anomaly Detection

Detects anomalies using reconstruction error from trained autoencoders.
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

# Try to import tensorflow/keras, but make it optional
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
        warnings.warn("TensorFlow/Keras not available. Autoencoder will use sklearn-based implementation.")

logger = logging.getLogger(__name__)


class AutoencoderDetector(BaseAnomalyDetector):
    """
    Autoencoder-based anomaly detector.

    Trains an autoencoder on normal data and uses reconstruction error
    as anomaly score. High reconstruction error indicates anomalies.
    """

    def __init__(
        self,
        encoding_dim: Optional[int] = None,
        hidden_layers: List[int] = None,
        activation: str = "relu",
        optimizer: str = "adam",
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        threshold_percentile: float = 95.0,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize Autoencoder detector.

        Args:
            encoding_dim: Dimension of encoding layer (auto if None)
            hidden_layers: List of hidden layer sizes (e.g., [64, 32, 16])
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            optimizer: Optimizer ('adam', 'sgd', 'rmsprop')
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            threshold_percentile: Percentile for threshold (default: 95th)
            config: Optional detector configuration
        """
        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)

        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.threshold_percentile = threshold_percentile

        self.model = None
        self.input_dim = None
        self.threshold_ = None
        self.use_tensorflow = TENSORFLOW_AVAILABLE

    def _build_model(self, input_dim: int):
        """Build autoencoder model."""
        if self.use_tensorflow:
            return self._build_keras_model(input_dim)
        else:
            return self._build_sklearn_model(input_dim)

    def _build_keras_model(self, input_dim: int):
        """Build Keras autoencoder model."""
        # Input layer
        input_layer = layers.Input(shape=(input_dim,))

        # Encoding layers
        encoded = input_layer
        for size in self.hidden_layers:
            encoded = layers.Dense(size, activation=self.activation)(encoded)

        # Encoding dimension
        if self.encoding_dim is None:
            encoding_dim = max(2, input_dim // 4)  # At least 2, or 1/4 of input
        else:
            encoding_dim = self.encoding_dim

        encoded = layers.Dense(encoding_dim, activation=self.activation, name="encoding")(encoded)

        # Decoding layers (reverse of encoding)
        decoded = encoded
        for size in reversed(self.hidden_layers):
            decoded = layers.Dense(size, activation=self.activation)(decoded)

        # Output layer
        decoded = layers.Dense(input_dim, activation="linear", name="decoded")(decoded)

        # Create model
        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer=self.optimizer, loss="mse", metrics=["mae"])

        return autoencoder

    def _build_sklearn_model(self, input_dim: int):
        """Build sklearn-based autoencoder (simplified)."""
        # Use PCA as a simple autoencoder alternative
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            if self.encoding_dim is None:
                encoding_dim = max(2, input_dim // 4)
            else:
                encoding_dim = self.encoding_dim

            pipeline = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=encoding_dim))])

            return pipeline
        except ImportError:
            raise ImportError("sklearn required for autoencoder when TensorFlow is not available")

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "AutoencoderDetector":
        """
        Fit the autoencoder on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for autoencoder (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)
        self.input_dim = array_data.shape[1]

        # Build model
        self.model = self._build_model(self.input_dim)

        # Train model
        if self.use_tensorflow:
            # Early stopping callback
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
            )

            # Train
            self.model.fit(
                array_data,
                array_data,  # Autoencoder: input = target
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=[early_stopping],
                verbose=0,
            )
        else:
            # sklearn-based training
            self.model.fit(array_data)

        # Calculate reconstruction errors on training data
        if self.use_tensorflow:
            reconstructed = self.model.predict(array_data, verbose=0)
            reconstruction_errors = np.mean((array_data - reconstructed) ** 2, axis=1)
        else:
            # For PCA-based, use reconstruction error
            # Scale first
            array_data_scaled = self.model.named_steps["scaler"].transform(array_data)
            transformed = self.model.named_steps["pca"].transform(array_data_scaled)
            reconstructed_scaled = self.model.named_steps["pca"].inverse_transform(transformed)
            reconstructed = self.model.named_steps["scaler"].inverse_transform(reconstructed_scaled)
            reconstruction_errors = np.mean((array_data - reconstructed) ** 2, axis=1)

        # Set threshold based on percentile
        self.threshold_ = np.percentile(reconstruction_errors, self.threshold_percentile)
        self.config.threshold = float(self.threshold_)

        self.is_fitted = True
        logger.info(f"AutoencoderDetector fitted on {array_data.shape[0]} samples. Threshold: {self.threshold_:.4f}")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using reconstruction error.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        # Check dimension before preprocessing to provide better error message
        # Convert to array first to check dimensions
        if isinstance(data, dict):
            array_data = self._dict_to_array(data)
        elif isinstance(data, pd.DataFrame):
            array_data = data.values
        else:
            array_data = np.asarray(data)

        if array_data.ndim == 1:
            array_data = array_data.reshape(1, -1)

        if array_data.shape[1] != self.input_dim:
            raise ValueError(f"dimension mismatch: expected {self.input_dim}, got {array_data.shape[1]}")

        array_data = self._preprocess_data(data)

        # Get reconstructions
        if self.use_tensorflow:
            reconstructed = self.model.predict(array_data, verbose=0)
            reconstruction_errors = np.mean((array_data - reconstructed) ** 2, axis=1)
        else:
            # Scale first
            array_data_scaled = self.model.named_steps["scaler"].transform(array_data)
            transformed = self.model.named_steps["pca"].transform(array_data_scaled)
            reconstructed_scaled = self.model.named_steps["pca"].inverse_transform(transformed)
            reconstructed = self.model.named_steps["scaler"].inverse_transform(reconstructed_scaled)
            reconstruction_errors = np.mean((array_data - reconstructed) ** 2, axis=1)

        # Anomaly scores are reconstruction errors
        anomaly_scores = reconstruction_errors

        # Get indices and coordinates if available
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(anomaly_scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(anomaly_scores)  # Placeholder

        # Create results
        results = self._create_results(scores=anomaly_scores, indices=indices, coordinates=coordinates)

        return results

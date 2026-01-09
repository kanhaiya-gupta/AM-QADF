"""
Variational Autoencoder (VAE) Based Anomaly Detection

Detects anomalies using reconstruction error and latent space distribution
from trained Variational Autoencoders.
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
    from tensorflow.keras import layers, models, backend as K

    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import keras
        from keras import layers, models
        from keras import backend as K

        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        warnings.warn("TensorFlow/Keras not available. VAEDetector requires TensorFlow/Keras.")

logger = logging.getLogger(__name__)


class VAEDetector(BaseAnomalyDetector):
    """
    Variational Autoencoder-based anomaly detector.

    Trains a VAE on normal data and uses reconstruction error combined
    with latent space distribution as anomaly score.
    """

    def __init__(
        self,
        encoding_dim: int = 16,
        hidden_layers: List[int] = None,
        activation: str = "relu",
        optimizer: str = "adam",
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        beta: float = 1.0,  # KL divergence weight
        threshold_percentile: float = 95.0,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize VAE detector.

        Args:
            encoding_dim: Dimension of latent space
            hidden_layers: List of hidden layer sizes (e.g., [64, 32])
            activation: Activation function ('relu', 'tanh')
            optimizer: Optimizer ('adam', 'sgd', 'rmsprop')
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            beta: Weight for KL divergence term (beta-VAE)
            threshold_percentile: Percentile for threshold (default: 95th)
            config: Optional detector configuration
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required for VAEDetector")

        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)

        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.beta = beta
        self.threshold_percentile = threshold_percentile

        self.encoder = None
        self.decoder = None
        self.vae = None
        self.input_dim = None
        self.threshold_ = None

    def _sampling(self, args):
        """Reparameterization trick for VAE."""
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def _build_model(self, input_dim: int):
        """Build VAE model."""
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))

        # Hidden layers
        x = input_layer
        for size in self.hidden_layers:
            x = layers.Dense(size, activation=self.activation)(x)

        # Latent space
        z_mean = layers.Dense(self.encoding_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.encoding_dim, name="z_log_var")(x)

        # Sampling layer
        z = layers.Lambda(self._sampling, output_shape=(self.encoding_dim,), name="z")([z_mean, z_log_var])

        # Encoder model
        encoder = models.Model(input_layer, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        latent_input = layers.Input(shape=(self.encoding_dim,), name="z_sampling")

        # Hidden layers (reverse)
        x = latent_input
        for size in reversed(self.hidden_layers):
            x = layers.Dense(size, activation=self.activation)(x)

        # Output layer
        output_layer = layers.Dense(input_dim, activation="linear", name="decoder_output")(x)

        # Decoder model
        decoder = models.Model(latent_input, output_layer, name="decoder")

        # VAE model
        encoder_outputs = encoder(input_layer)
        z_mean, z_log_var, z = encoder_outputs
        output = decoder(z)

        # VAE model - simple output for Keras 3.x compatibility
        vae = models.Model(input_layer, output, name="vae")

        # Try to use add_loss for Keras 2.x, fallback to MSE for Keras 3.x
        try:
            # Compute VAE loss
            try:
                import keras.ops as ops

                recon_loss = ops.mean(ops.square(input_layer - output), axis=-1)
                recon_loss = ops.mean(recon_loss) * input_dim
                kl_loss = 1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var)
                kl_loss = ops.sum(kl_loss, axis=-1)
                kl_loss = ops.mean(kl_loss) * -0.5 * self.beta
                total_loss = recon_loss + kl_loss
            except (ImportError, AttributeError):
                recon_loss = K.mean(K.square(input_layer - output), axis=-1)
                recon_loss = K.mean(recon_loss) * input_dim
                kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
                kl_loss = K.sum(kl_loss, axis=-1)
                kl_loss = K.mean(kl_loss) * -0.5 * self.beta
                total_loss = recon_loss + kl_loss

            # Try add_loss (works in Keras 2.x, not in 3.x)
            vae.add_loss(total_loss)
            vae.compile(optimizer=self.optimizer)
        except NotImplementedError:
            # Keras 3.x: add_loss() is not supported
            # Use MSE loss as fallback (KL loss will be approximated via regularization)
            # Note: This is a simplified VAE that focuses on reconstruction
            vae.compile(optimizer=self.optimizer, loss="mse")
            logger.warning("Keras 3.x detected: Using MSE loss instead of full VAE loss. KL divergence term is not included.")
        except Exception as e:
            # Final fallback: simple MSE loss
            vae.compile(optimizer=self.optimizer, loss="mse")
            logger.warning(f"Could not set up VAE loss: {e}. Using MSE loss as fallback.")

        return encoder, decoder, vae

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "VAEDetector":
        """
        Fit the VAE on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for VAE (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)
        self.input_dim = array_data.shape[1]

        # Build model
        self.encoder, self.decoder, self.vae = self._build_model(self.input_dim)

        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)

        # Train
        self.vae.fit(
            array_data,
            array_data,  # VAE: input = target
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping],
            verbose=0,
        )

        # Calculate reconstruction errors on training data
        z_mean, z_log_var, z = self.encoder.predict(array_data, verbose=0)
        reconstructed = self.decoder.predict(z, verbose=0)
        reconstruction_errors = np.mean((array_data - reconstructed) ** 2, axis=1)

        # Also consider KL divergence as part of anomaly score (with numerical stability)
        # Clip z_log_var to prevent overflow/underflow in exp
        z_log_var_clipped = np.clip(z_log_var, -10, 10)
        kl_divergences = -0.5 * np.sum(
            1 + z_log_var_clipped - np.square(z_mean) - np.exp(z_log_var_clipped),
            axis=1,
        )

        # Ensure KL divergences are finite
        kl_divergences = np.nan_to_num(kl_divergences, nan=0.0, posinf=1e6, neginf=-1e6)

        # Combined anomaly score
        anomaly_scores = reconstruction_errors + self.beta * kl_divergences

        # Ensure all scores are finite
        anomaly_scores = np.nan_to_num(
            anomaly_scores,
            nan=0.0,
            posinf=(np.max(anomaly_scores[np.isfinite(anomaly_scores)]) if np.any(np.isfinite(anomaly_scores)) else 1e6),
            neginf=0.0,
        )

        # Set threshold based on percentile
        self.threshold_ = np.percentile(anomaly_scores, self.threshold_percentile)
        self.config.threshold = float(self.threshold_)

        self.is_fitted = True
        logger.info(f"VAEDetector fitted on {array_data.shape[0]} samples. Threshold: {self.threshold_:.4f}")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using VAE reconstruction error and KL divergence.

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

        # Get latent representations
        z_mean, z_log_var, z = self.encoder.predict(array_data, verbose=0)

        # Get reconstructions
        reconstructed = self.decoder.predict(z, verbose=0)
        reconstruction_errors = np.mean((array_data - reconstructed) ** 2, axis=1)

        # Calculate KL divergence with numerical stability
        # Clip z_log_var to prevent overflow/underflow in exp
        z_log_var_clipped = np.clip(z_log_var, -10, 10)
        kl_divergences = -0.5 * np.sum(
            1 + z_log_var_clipped - np.square(z_mean) - np.exp(z_log_var_clipped),
            axis=1,
        )

        # Ensure KL divergences are finite
        kl_divergences = np.nan_to_num(kl_divergences, nan=0.0, posinf=1e6, neginf=-1e6)

        # Combined anomaly score
        anomaly_scores = reconstruction_errors + self.beta * kl_divergences

        # Ensure all scores are finite (replace NaN/inf with 0 or max value)
        anomaly_scores = np.nan_to_num(
            anomaly_scores,
            nan=0.0,
            posinf=(np.max(anomaly_scores[np.isfinite(anomaly_scores)]) if np.any(np.isfinite(anomaly_scores)) else 1e6),
            neginf=0.0,
        )

        # Get indices and coordinates if available
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(anomaly_scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(anomaly_scores)  # Placeholder

        # Create results
        results = self._create_results(scores=anomaly_scores, indices=indices, coordinates=coordinates)

        return results

"""
Machine Learning Based Anomaly Detection Methods

This module provides deep learning and ensemble-based anomaly detection methods
for PBF-LB/M process analysis.
"""

from .autoencoder import AutoencoderDetector
from .random_forest import RandomForestAnomalyDetector

# LSTM and VAE are optional (require tensorflow/pytorch)
try:
    from .lstm_autoencoder import LSTMAutoencoderDetector

    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    LSTMAutoencoderDetector = None

try:
    from .vae import VAEDetector

    VAE_AVAILABLE = True
except ImportError:
    VAE_AVAILABLE = False
    VAEDetector = None

__all__ = [
    "AutoencoderDetector",
    "RandomForestAnomalyDetector",
]

if LSTM_AVAILABLE:
    __all__.append("LSTMAutoencoderDetector")
if VAE_AVAILABLE:
    __all__.append("VAEDetector")

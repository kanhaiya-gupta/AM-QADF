"""
Ensemble Anomaly Detection Methods

This module provides ensemble methods that combine multiple detection
algorithms for improved performance.
"""

from .voting_ensemble import VotingEnsembleDetector
from .weighted_ensemble import WeightedEnsembleDetector

__all__ = ["VotingEnsembleDetector", "WeightedEnsembleDetector"]

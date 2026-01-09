"""
Unit tests for Voting Ensemble detector.

Tests for VotingEnsembleDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from am_qadf.anomaly_detection.detectors.ensemble.voting_ensemble import (
    VotingEnsembleDetector,
)
from am_qadf.anomaly_detection.core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)
from am_qadf.anomaly_detection.core.types import AnomalyType


class MockDetector(BaseAnomalyDetector):
    """Mock detector for testing ensemble."""

    def __init__(self, name="MockDetector", predict_anomaly=False):
        super().__init__()
        self.name = name
        self.predict_anomaly = predict_anomaly
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
                    is_anomaly=self.predict_anomaly,
                    anomaly_score=0.8 if self.predict_anomaly else 0.2,
                    confidence=0.9,
                    detector_name=self.name,
                )
            )
        return results


class TestVotingEnsembleDetector:
    """Test suite for VotingEnsembleDetector class."""

    @pytest.fixture
    def mock_detectors(self):
        """Create mock detectors for testing."""
        return [
            MockDetector(name="Detector1", predict_anomaly=False),
            MockDetector(name="Detector2", predict_anomaly=False),
            MockDetector(name="Detector3", predict_anomaly=True),
        ]

    @pytest.fixture
    def normal_data(self):
        """Create normal data for training."""
        np.random.seed(42)
        return np.random.randn(50, 3) * 10 + 100

    @pytest.mark.unit
    def test_detector_initialization_default(self, mock_detectors):
        """Test VotingEnsembleDetector initialization with default values."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors)

        assert len(ensemble.detectors) == 3
        assert ensemble.voting_method == "majority"
        assert ensemble.threshold is None
        assert ensemble.config is not None

    @pytest.mark.unit
    def test_detector_initialization_majority(self, mock_detectors):
        """Test initialization with majority voting."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors, voting_method="majority")

        assert ensemble.voting_method == "majority"
        assert ensemble.threshold is None

    @pytest.mark.unit
    def test_detector_initialization_unanimous(self, mock_detectors):
        """Test initialization with unanimous voting."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors, voting_method="unanimous")

        assert ensemble.voting_method == "unanimous"
        assert ensemble.threshold is None

    @pytest.mark.unit
    def test_detector_initialization_at_least_n(self, mock_detectors):
        """Test initialization with at_least_n voting."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors, voting_method="at_least_n", threshold=2)

        assert ensemble.voting_method == "at_least_n"
        assert ensemble.threshold == 2

    @pytest.mark.unit
    def test_detector_initialization_at_least_n_default(self, mock_detectors):
        """Test initialization with at_least_n voting and default threshold."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors, voting_method="at_least_n")

        # Default threshold should be len(detectors) // 2 + 1
        assert ensemble.threshold == len(mock_detectors) // 2 + 1

    @pytest.mark.unit
    def test_detector_initialization_with_config(self, mock_detectors):
        """Test initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        ensemble = VotingEnsembleDetector(detectors=mock_detectors, config=config)

        assert ensemble.config.threshold == 0.5

    @pytest.mark.unit
    def test_fit(self, mock_detectors, normal_data):
        """Test fitting the ensemble."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors)
        ensemble.fit(normal_data)

        assert ensemble.is_fitted is True
        # All detectors should be fitted
        assert all(d.fit_called for d in mock_detectors)
        assert all(d.is_fitted for d in mock_detectors)

    @pytest.mark.unit
    def test_fit_with_labels(self, mock_detectors, normal_data):
        """Test fitting with labels."""
        labels = np.random.randint(0, 2, len(normal_data))
        ensemble = VotingEnsembleDetector(detectors=mock_detectors)

        ensemble.fit(normal_data, labels)

        assert ensemble.is_fitted is True

    @pytest.mark.unit
    def test_predict_before_fit(self, mock_detectors, normal_data):
        """Test that predict raises error if not fitted."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors)

        with pytest.raises(ValueError, match="must be fitted"):
            ensemble.predict(normal_data)

    @pytest.mark.unit
    def test_predict_majority_voting(self, mock_detectors, normal_data):
        """Test prediction with majority voting."""
        # 2 out of 3 detectors say no anomaly, 1 says anomaly
        # Majority should be no anomaly
        ensemble = VotingEnsembleDetector(detectors=mock_detectors, voting_method="majority")
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        assert len(results) == 1
        # Majority (2/3) say no anomaly
        assert results[0].is_anomaly is False
        assert results[0].detector_name == "VotingEnsemble(majority)"
        assert "votes" in results[0].metadata
        assert results[0].metadata["voting_method"] == "majority"
        assert results[0].metadata["n_detectors"] == 3

    @pytest.mark.unit
    def test_predict_majority_voting_anomaly(self, normal_data):
        """Test majority voting when majority says anomaly."""
        # Create detectors where 2 out of 3 say anomaly
        detectors = [
            MockDetector(name="Detector1", predict_anomaly=True),
            MockDetector(name="Detector2", predict_anomaly=True),
            MockDetector(name="Detector3", predict_anomaly=False),
        ]

        ensemble = VotingEnsembleDetector(detectors=detectors, voting_method="majority")
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # Majority (2/3) say anomaly
        assert results[0].is_anomaly is True

    @pytest.mark.unit
    def test_predict_unanimous_voting(self, mock_detectors, normal_data):
        """Test prediction with unanimous voting."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors, voting_method="unanimous")
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # Not all agree (2 say no, 1 says yes)
        assert results[0].is_anomaly is False

    @pytest.mark.unit
    def test_predict_unanimous_voting_all_agree(self, normal_data):
        """Test unanimous voting when all agree."""
        # All detectors say anomaly
        detectors = [
            MockDetector(name="Detector1", predict_anomaly=True),
            MockDetector(name="Detector2", predict_anomaly=True),
            MockDetector(name="Detector3", predict_anomaly=True),
        ]

        ensemble = VotingEnsembleDetector(detectors=detectors, voting_method="unanimous")
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # All agree on anomaly
        assert results[0].is_anomaly is True

    @pytest.mark.unit
    def test_predict_at_least_n_voting(self, mock_detectors, normal_data):
        """Test prediction with at_least_n voting."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors, voting_method="at_least_n", threshold=2)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # Only 1 detector says anomaly, need at least 2
        assert results[0].is_anomaly is False

    @pytest.mark.unit
    def test_predict_at_least_n_voting_meets_threshold(self, normal_data):
        """Test at_least_n voting when threshold is met."""
        # 2 out of 3 detectors say anomaly
        detectors = [
            MockDetector(name="Detector1", predict_anomaly=True),
            MockDetector(name="Detector2", predict_anomaly=True),
            MockDetector(name="Detector3", predict_anomaly=False),
        ]

        ensemble = VotingEnsembleDetector(detectors=detectors, voting_method="at_least_n", threshold=2)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # 2 detectors say anomaly, threshold is 2
        assert results[0].is_anomaly is True

    @pytest.mark.unit
    def test_predict_average_score(self, mock_detectors, normal_data):
        """Test that scores are averaged."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # Scores: 0.2, 0.2, 0.8 -> average = 0.4
        expected_score = (0.2 + 0.2 + 0.8) / 3
        assert abs(results[0].anomaly_score - expected_score) < 1e-10

    @pytest.mark.unit
    def test_predict_multiple_samples(self, mock_detectors, normal_data):
        """Test prediction on multiple samples."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:5])

        assert len(results) == 5
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_predict_metadata(self, mock_detectors, normal_data):
        """Test that metadata includes vote information."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors)
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        assert "votes" in results[0].metadata
        assert len(results[0].metadata["votes"]) == 3
        assert results[0].metadata["votes"] == [False, False, True]
        assert results[0].metadata["voting_method"] == "majority"
        assert results[0].metadata["n_detectors"] == 3

    @pytest.mark.unit
    def test_predict_invalid_voting_method(self, mock_detectors, normal_data):
        """Test that invalid voting method defaults to majority."""
        ensemble = VotingEnsembleDetector(detectors=mock_detectors, voting_method="invalid_method")
        ensemble.fit(normal_data)

        # Should not raise error, defaults to majority
        results = ensemble.predict(normal_data[:1])
        assert len(results) == 1

    @pytest.mark.unit
    def test_edge_case_single_detector(self, normal_data):
        """Test with single detector."""
        detector = MockDetector(name="SingleDetector", predict_anomaly=True)
        ensemble = VotingEnsembleDetector(detectors=[detector])
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        assert len(results) == 1
        # Single detector's vote is the ensemble vote
        assert results[0].is_anomaly is True

    @pytest.mark.unit
    def test_edge_case_tie_vote(self, normal_data):
        """Test with tie vote (even number of detectors)."""
        # 2 detectors, one says anomaly, one says no
        detectors = [
            MockDetector(name="Detector1", predict_anomaly=True),
            MockDetector(name="Detector2", predict_anomaly=False),
        ]

        ensemble = VotingEnsembleDetector(detectors=detectors, voting_method="majority")
        ensemble.fit(normal_data)

        results = ensemble.predict(normal_data[:1])

        # Tie: 1 yes, 1 no -> sum(votes) = 1, len(votes)/2 = 1
        # sum(votes) > len(votes)/2 is False, so no anomaly
        assert results[0].is_anomaly is False

    @pytest.mark.unit
    def test_edge_case_empty_detectors(self):
        """Test that empty detector list raises error."""
        with pytest.raises((ValueError, IndexError, AttributeError)):
            # This might fail at different points depending on implementation
            ensemble = VotingEnsembleDetector(detectors=[])
            # Or might fail during fit/predict
            ensemble.fit(np.random.randn(10, 3))

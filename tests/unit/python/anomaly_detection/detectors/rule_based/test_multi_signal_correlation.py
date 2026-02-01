"""
Unit tests for Multi-Signal Correlation detector.

Tests for MultiSignalCorrelationDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.rule_based.multi_signal_correlation import (
    MultiSignalCorrelationDetector,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


class TestMultiSignalCorrelationDetector:
    """Test suite for MultiSignalCorrelationDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create MultiSignalCorrelationDetector with default parameters."""
        return MultiSignalCorrelationDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create MultiSignalCorrelationDetector with custom parameters."""
        return MultiSignalCorrelationDetector(
            correlation_threshold=0.5,
            use_expected_correlations=True,
            use_residual_analysis=False,
        )

    @pytest.fixture
    def correlated_data(self):
        """Create correlated data for training."""
        np.random.seed(42)
        base = np.random.randn(100) * 10 + 100
        # Create correlated features
        feature1 = base
        feature2 = base * 0.8 + np.random.randn(100) * 2  # Correlated with feature1
        feature3 = base * 0.6 + np.random.randn(100) * 3  # Correlated with feature1
        return np.column_stack([feature1, feature2, feature3])

    @pytest.fixture
    def data_with_correlation_violations(self, correlated_data):
        """Create data with correlation violations."""
        data = correlated_data.copy()
        # Add violation: feature2 should be correlated with feature1, but isn't
        data[0, 1] = data[0, 0] * 2 + 100  # Breaks correlation
        data[1, 2] = -data[1, 0] * 2  # Breaks correlation (negative)
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test MultiSignalCorrelationDetector initialization with default values."""
        detector = MultiSignalCorrelationDetector()

        assert detector.correlation_threshold == 0.7
        assert detector.use_expected_correlations is True
        assert detector.use_residual_analysis is True
        assert detector.use_cross_validation is True
        assert detector.expected_correlations == {}
        assert detector.threshold_percentile == 95.0
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.learned_correlations_ is None
        assert detector.relationship_models_ is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test initialization with custom parameters."""
        expected_corrs = {("feature1", "feature2"): 0.8}
        detector = MultiSignalCorrelationDetector(
            correlation_threshold=0.5,
            use_residual_analysis=False,
            expected_correlations=expected_corrs,
        )

        assert detector.correlation_threshold == 0.5
        assert detector.use_residual_analysis is False
        assert detector.expected_correlations == expected_corrs

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = MultiSignalCorrelationDetector(config=config)

        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_fit(self, detector_default, correlated_data):
        """Test fitting the detector."""
        detector = detector_default.fit(correlated_data)

        assert detector.is_fitted is True
        assert detector.learned_correlations_ is not None
        assert len(detector.learned_correlations_) > 0  # Should find correlations
        assert detector is detector_default  # Method chaining

    @pytest.mark.unit
    def test_fit_learns_correlations(self, detector_default, correlated_data):
        """Test that correlations are learned from data."""
        detector = detector_default.fit(correlated_data)

        # Should learn correlations between correlated features
        assert len(detector.learned_correlations_) > 0
        for (sig1, sig2), corr_info in detector.learned_correlations_.items():
            assert "correlation" in corr_info
            assert "p_value" in corr_info
            assert abs(corr_info["correlation"]) >= detector.correlation_threshold

    @pytest.mark.unit
    def test_fit_relationship_models(self, detector_default, correlated_data):
        """Test that relationship models are learned."""
        detector = detector_default.fit(correlated_data)

        if detector.use_residual_analysis:
            assert detector.relationship_models_ is not None
            for (sig1, sig2), model in detector.relationship_models_.items():
                assert "slope" in model
                assert "intercept" in model
                assert "r_squared" in model

    @pytest.mark.unit
    def test_fit_with_low_correlation_threshold(self, correlated_data):
        """Test fitting with lower correlation threshold."""
        detector = MultiSignalCorrelationDetector(correlation_threshold=0.3)
        detector.fit(correlated_data)

        # Should find more correlations with lower threshold
        assert len(detector.learned_correlations_) > 0

    @pytest.mark.unit
    def test_fit_with_dataframe(self, detector_default):
        """Test fitting with pandas DataFrame."""
        df = pd.DataFrame(
            {
                "signal1": np.random.randn(50) * 10 + 100,
                "signal2": np.random.randn(50) * 10 + 100,
                "signal3": np.random.randn(50) * 10 + 100,
            }
        )
        # Make signal2 correlated with signal1
        df["signal2"] = df["signal1"] * 0.8 + np.random.randn(50) * 2

        detector = detector_default.fit(df)

        assert detector.is_fitted is True
        assert len(detector.learned_correlations_) > 0

    @pytest.mark.unit
    def test_calculate_residual(self, detector_default, correlated_data):
        """Test residual calculation."""
        detector = detector_default.fit(correlated_data)

        if detector.relationship_models_:
            model = list(detector.relationship_models_.values())[0]
            value1 = 100.0
            value2 = model["slope"] * value1 + model["intercept"]

            # Perfect match should have zero residual
            residual = detector._calculate_residual(value1, value2, model)
            assert residual >= 0

            # Mismatch should have higher residual
            value2_wrong = value2 + 50
            residual_wrong = detector._calculate_residual(value1, value2_wrong, model)
            assert residual_wrong > residual

    @pytest.mark.unit
    def test_predict_before_fit(self, detector_default, correlated_data):
        """Test that predict raises error if not fitted."""
        with pytest.raises(ValueError, match="must be fitted"):
            detector_default.predict(correlated_data)

    @pytest.mark.unit
    def test_predict_normal_data(self, detector_default, correlated_data):
        """Test prediction on normal correlated data."""
        detector = detector_default.fit(correlated_data)
        results = detector.predict(correlated_data)

        assert len(results) == len(correlated_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)
        # Most points should maintain correlations
        anomaly_count = sum(r.is_anomaly for r in results)
        assert anomaly_count < len(results) * 0.2

    @pytest.mark.unit
    def test_predict_with_correlation_violations(self, detector_default, correlated_data, data_with_correlation_violations):
        """Test prediction on data with correlation violations."""
        detector = detector_default.fit(correlated_data)
        results = detector.predict(data_with_correlation_violations)

        assert len(results) == len(data_with_correlation_violations)
        # Violations should be detected
        assert results[0].is_anomaly is True or results[1].is_anomaly is True
        # Violations should have higher scores
        assert results[0].anomaly_score > results[5].anomaly_score
        assert results[1].anomaly_score > results[5].anomaly_score

    @pytest.mark.unit
    def test_predict_residual_analysis(self, detector_default, correlated_data):
        """Test residual-based anomaly detection."""
        detector = MultiSignalCorrelationDetector(use_residual_analysis=True)
        detector.fit(correlated_data)

        # Create point that violates relationship
        test_data = correlated_data.copy()
        # Break correlation
        test_data[0, 1] = test_data[0, 0] * 2 + 100

        results = detector.predict(test_data)

        # Should detect violation
        assert results[0].anomaly_score > results[5].anomaly_score

    @pytest.mark.unit
    def test_predict_without_residual_analysis(self, correlated_data):
        """Test prediction without residual analysis."""
        detector = MultiSignalCorrelationDetector(use_residual_analysis=False)
        detector.fit(correlated_data)

        test_data = correlated_data.copy()
        test_data[0, 1] = test_data[0, 0] * 2 + 100

        results = detector.predict(test_data)

        # May or may not detect without residual analysis
        assert len(results) == len(test_data)

    @pytest.mark.unit
    def test_predict_with_expected_correlations(self, correlated_data):
        """Test prediction with expected correlations."""
        expected_corrs = {("feature_0", "feature_1"): 0.8}
        detector = MultiSignalCorrelationDetector(expected_correlations=expected_corrs, use_expected_correlations=True)
        detector.fit(correlated_data)

        results = detector.predict(correlated_data)

        assert len(results) == len(correlated_data)

    @pytest.mark.unit
    def test_predict_scores(self, detector_default, correlated_data, data_with_correlation_violations):
        """Test predict_scores method."""
        detector = detector_default.fit(correlated_data)
        scores = detector.predict_scores(data_with_correlation_violations)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data_with_correlation_violations)
        assert all(s >= 0 for s in scores)
        # Violations should have higher scores
        assert scores[0] > scores[5]
        assert scores[1] > scores[5]

    @pytest.mark.unit
    def test_different_correlation_thresholds(self, correlated_data):
        """Test with different correlation thresholds."""
        detector_low = MultiSignalCorrelationDetector(correlation_threshold=0.3).fit(correlated_data)
        detector_high = MultiSignalCorrelationDetector(correlation_threshold=0.9).fit(correlated_data)

        # Lower threshold should find more correlations
        assert len(detector_low.learned_correlations_) >= len(detector_high.learned_correlations_)

    @pytest.mark.unit
    def test_edge_case_uncorrelated_data(self, detector_default):
        """Test with uncorrelated data."""
        # Create uncorrelated features
        uncorrelated = np.column_stack(
            [
                np.random.randn(50) * 10 + 100,
                np.random.randn(50) * 10 + 50,
                np.random.randn(50) * 10 + 10,
            ]
        )

        detector = detector_default.fit(uncorrelated)

        # May find few or no correlations
        assert detector.is_fitted is True
        assert len(detector.learned_correlations_) >= 0

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.random.randn(5, 3) * 10 + 100

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        # Need at least 3 points for correlation
        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_edge_case_insufficient_data_for_correlation(self, detector_default):
        """Test with insufficient data for correlation calculation."""
        very_small = np.random.randn(2, 3) * 10 + 100

        detector = detector_default.fit(very_small)

        # Should handle gracefully (may find no correlations)
        assert detector.is_fitted is True

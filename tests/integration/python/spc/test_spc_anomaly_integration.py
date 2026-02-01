"""
Integration tests for SPC with PatternDeviationDetector.

Tests integration of SPC module with anomaly detection.
"""

import pytest
import numpy as np

try:
    from am_qadf.anomaly_detection.detectors.rule_based.pattern_deviation import PatternDeviationDetector
    from am_qadf.anomaly_detection.core.base_detector import AnomalyDetectionConfig
    from am_qadf.analytics.spc import SPCConfig
except ImportError:
    pytest.skip("SPC or anomaly detection modules not available", allow_module_level=True)


class TestSPCAnomalyIntegration:
    """Integration tests for SPC with PatternDeviationDetector."""

    @pytest.fixture
    def detector_with_spc(self):
        """PatternDeviationDetector with SPC module enabled."""
        return PatternDeviationDetector(control_limit_sigma=3.0, pattern_window=10, use_spc_module=True)

    @pytest.fixture
    def detector_without_spc(self):
        """PatternDeviationDetector with legacy implementation."""
        return PatternDeviationDetector(control_limit_sigma=3.0, pattern_window=10, use_spc_module=False)

    @pytest.mark.integration
    def test_detector_with_spc_enabled(self, detector_with_spc):
        """Test PatternDeviationDetector with SPC module enabled."""
        assert detector_with_spc.use_spc_module == True
        # spc_client may be None if module unavailable
        if detector_with_spc.spc_client is not None:
            assert detector_with_spc.spc_config is not None

    @pytest.mark.integration
    def test_detector_fit_with_spc(self, detector_with_spc):
        """Test fitting detector with SPC module."""
        # Generate training data
        training_data = np.random.randn(100, 3)  # 100 samples, 3 features

        if detector_with_spc.use_spc_module and detector_with_spc.spc_client is not None:
            detector_with_spc.fit(training_data)

            assert detector_with_spc.is_fitted == True
            assert detector_with_spc.control_limits_ is not None
            assert detector_with_spc.baseline_stats_ is not None
        else:
            pytest.skip("SPC module not available")

    @pytest.mark.integration
    def test_detector_predict_with_spc(self, detector_with_spc):
        """Test predicting with SPC module."""
        # Generate training and test data
        training_data = np.random.randn(100, 3)
        test_data = np.random.randn(50, 3)
        # Add some anomalies
        test_data[10:15] += 5.0  # Shift some points

        if detector_with_spc.use_spc_module and detector_with_spc.spc_client is not None:
            detector_with_spc.fit(training_data)
            results = detector_with_spc.predict(test_data)

            assert isinstance(results, list)
            assert len(results) > 0
            # Should detect anomalies
            anomaly_scores = [r.anomaly_score for r in results]
            assert max(anomaly_scores) > min(anomaly_scores)  # Some variation
        else:
            pytest.skip("SPC module not available")

    @pytest.mark.integration
    def test_detector_backward_compatibility(self, detector_without_spc):
        """Test backward compatibility with legacy implementation."""
        training_data = np.random.randn(100, 3)
        test_data = np.random.randn(50, 3)

        detector_without_spc.fit(training_data)
        results = detector_without_spc.predict(test_data)

        assert detector_without_spc.is_fitted == True
        assert isinstance(results, list)
        assert len(results) == len(test_data)

    @pytest.mark.integration
    def test_detector_rule_violations_as_anomalies(self, detector_with_spc):
        """Test that rule violations are detected as anomalies."""
        training_data = np.random.randn(100, 3)
        # Create test data with trend (should trigger rule violations)
        test_data = np.zeros((50, 3))
        for i in range(50):
            test_data[i] = np.array([10.0 + i * 0.1, 10.0 + i * 0.1, 10.0 + i * 0.1])  # Trend

        if detector_with_spc.use_spc_module and detector_with_spc.spc_client is not None:
            detector_with_spc.fit(training_data)
            results = detector_with_spc.predict(test_data)

            assert isinstance(results, list)
            # Should detect trend as anomalies
            assert any(r.anomaly_score > 0 for r in results)
        else:
            pytest.skip("SPC module not available")

    @pytest.mark.integration
    def test_detector_out_of_control_points(self, detector_with_spc):
        """Test that out-of-control points are detected as anomalies."""
        training_data = np.random.randn(100, 3)
        test_data = np.random.randn(50, 3)
        # Add significant outliers
        test_data[10] = np.array([20.0, 20.0, 20.0])  # Far out of control
        test_data[25] = np.array([-10.0, -10.0, -10.0])  # Far out of control

        if detector_with_spc.use_spc_module and detector_with_spc.spc_client is not None:
            detector_with_spc.fit(training_data)
            results = detector_with_spc.predict(test_data)

            assert isinstance(results, list)
            # Out-of-control points should have high scores
            scores = [r.anomaly_score for r in results]
            assert scores[10] > np.percentile(scores, 95)  # Should be in top 5%
            assert scores[25] > np.percentile(scores, 95)
        else:
            pytest.skip("SPC module not available")

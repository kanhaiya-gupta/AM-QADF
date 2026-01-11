"""
Unit tests for multivariate SPC.

Tests for MultivariateSPCAnalyzer, Hotelling T², and PCA-based SPC.
"""

import pytest
import numpy as np

from am_qadf.analytics.spc.multivariate_spc import (
    MultivariateSPCAnalyzer,
    MultivariateSPCResult,
)
from am_qadf.analytics.spc.spc_client import SPCConfig
from tests.fixtures.spc.multivariate_data import (
    generate_multivariate_in_control_data,
    generate_multivariate_out_of_control_data,
    generate_high_dimensional_data,
)


class TestMultivariateSPCResult:
    """Test suite for MultivariateSPCResult dataclass."""

    @pytest.mark.unit
    def test_multivariate_spc_result_creation(self):
        """Test creating MultivariateSPCResult."""
        result = MultivariateSPCResult(
            hotelling_t2=np.array([1.0, 2.0, 3.0]),
            ucl_t2=10.0,
            control_limits={"variable_0": {"mean": 0.0, "ucl": 3.0, "lcl": -3.0}},
            out_of_control_points=[2],
            principal_components=None,
            explained_variance=None,
            contribution_analysis=None,
            baseline_mean=np.array([0.0, 0.0]),
            baseline_covariance=np.eye(2),
            metadata={"n_variables": 2},
        )

        assert len(result.hotelling_t2) == 3
        assert result.ucl_t2 == 10.0
        assert len(result.out_of_control_points) == 1
        assert result.baseline_mean.shape == (2,)
        assert result.baseline_covariance.shape == (2, 2)


class TestMultivariateSPCAnalyzer:
    """Test suite for MultivariateSPCAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a MultivariateSPCAnalyzer instance."""
        return MultivariateSPCAnalyzer()

    @pytest.fixture
    def config(self):
        """Create an SPCConfig instance."""
        return SPCConfig(control_limit_sigma=3.0)

    @pytest.mark.unit
    def test_analyzer_creation(self, analyzer):
        """Test creating MultivariateSPCAnalyzer."""
        assert analyzer is not None

    @pytest.mark.unit
    def test_create_hotelling_t2_chart(self, analyzer, config):
        """Test creating Hotelling T² chart."""
        data, cov = generate_multivariate_in_control_data(n_samples=100, n_variables=5)

        result = analyzer.create_hotelling_t2_chart(data, config=config, alpha=0.05)

        assert isinstance(result, MultivariateSPCResult)
        assert len(result.hotelling_t2) == len(data)
        assert result.ucl_t2 > 0
        assert result.baseline_mean.shape == (5,)
        assert result.baseline_covariance.shape == (5, 5)
        assert isinstance(result.out_of_control_points, list)

    @pytest.mark.unit
    def test_create_hotelling_t2_chart_with_baseline(self, analyzer, config):
        """Test creating Hotelling T² chart with provided baseline."""
        baseline_data, _ = generate_multivariate_in_control_data(n_samples=80, n_variables=5, seed=42)
        monitoring_data, _ = generate_multivariate_in_control_data(n_samples=50, n_variables=5, seed=43)

        result = analyzer.create_hotelling_t2_chart(monitoring_data, baseline_data=baseline_data, config=config)

        assert isinstance(result, MultivariateSPCResult)
        assert len(result.hotelling_t2) == len(monitoring_data)

    @pytest.mark.unit
    def test_create_hotelling_t2_chart_insufficient_data(self, analyzer, config):
        """Test creating Hotelling T² chart with insufficient data."""
        data = np.random.randn(3, 5)  # Need at least n_variables + 1 samples

        with pytest.raises(ValueError, match="Need at least"):
            analyzer.create_hotelling_t2_chart(data, config=config)

    @pytest.mark.unit
    def test_create_hotelling_t2_chart_too_few_variables(self, analyzer, config):
        """Test creating Hotelling T² chart with too few variables."""
        data = np.random.randn(50, 1)  # Need at least 2 variables

        with pytest.raises(ValueError, match="Multivariate SPC requires at least 2 variables"):
            analyzer.create_hotelling_t2_chart(data, config=config)

    @pytest.mark.unit
    def test_create_hotelling_t2_chart_out_of_control(self, analyzer, config):
        """Test detecting out-of-control points with Hotelling T²."""
        data, cov = generate_multivariate_out_of_control_data(n_samples=100, n_variables=5, shift_at=50, shift_magnitude=3.0)

        result = analyzer.create_hotelling_t2_chart(data, config=config)

        assert len(result.out_of_control_points) > 0
        # Should detect shift after index 50
        ooc_after_shift = [i for i in result.out_of_control_points if i >= 50]
        assert len(ooc_after_shift) > 0

    @pytest.mark.unit
    def test_create_pca_chart(self, analyzer, config):
        """Test creating PCA-based multivariate chart."""
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        data, cov = generate_multivariate_in_control_data(n_samples=100, n_variables=10)

        result = analyzer.create_pca_chart(data, n_components=3, variance_threshold=0.95, config=config)

        assert isinstance(result, MultivariateSPCResult)
        assert result.principal_components is not None
        assert result.explained_variance is not None
        assert len(result.explained_variance) <= 10

    @pytest.mark.unit
    def test_create_pca_chart_auto_components(self, analyzer, config):
        """Test creating PCA chart with automatic component selection."""
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        data, cov = generate_multivariate_in_control_data(n_samples=100, n_variables=10)

        result = analyzer.create_pca_chart(data, n_components=None, variance_threshold=0.95, config=config)  # Auto-select

        assert isinstance(result, MultivariateSPCResult)
        assert result.principal_components is not None
        assert result.explained_variance is not None
        # Should select components to explain >= 95% variance
        assert np.sum(result.explained_variance) >= 0.95

    @pytest.mark.unit
    def test_create_pca_chart_no_sklearn(self, analyzer, config):
        """Test creating PCA chart when scikit-learn is not available."""
        # Temporarily disable sklearn
        original_available = analyzer.sklearn_available
        analyzer.sklearn_available = False

        data, cov = generate_multivariate_in_control_data(n_samples=100, n_variables=5)

        with pytest.raises(ImportError, match="scikit-learn is required"):
            analyzer.create_pca_chart(data, config=config)

        # Restore
        analyzer.sklearn_available = original_available

    @pytest.mark.unit
    def test_calculate_contribution(self, analyzer, config):
        """Test calculating variable contributions to out-of-control point."""
        data, cov = generate_multivariate_in_control_data(n_samples=100, n_variables=5)
        result = analyzer.create_hotelling_t2_chart(data, config=config)

        if len(result.out_of_control_points) > 0:
            ooc_index = result.out_of_control_points[0]
            contributions = analyzer.calculate_contribution(
                result, ooc_index, data, variable_names=[f"var_{i}" for i in range(5)]
            )

            assert isinstance(contributions, dict)
            assert len(contributions) == 5
            # Contributions should sum to approximately 1.0 (may vary due to normalization)
            total = sum(contributions.values())
            assert 0.8 <= total <= 1.2  # Allow some tolerance

    @pytest.mark.unit
    def test_detect_multivariate_outliers(self, analyzer):
        """Test detecting multivariate outliers."""
        data, cov = generate_multivariate_in_control_data(n_samples=80, n_variables=5, seed=42)

        baseline_mean = np.mean(data, axis=0)
        baseline_cov = np.cov(data, rowvar=False, ddof=1)

        # Add some outliers
        test_data = data.copy()
        test_data[10] = baseline_mean + 5.0 * np.sqrt(np.diag(baseline_cov))
        test_data[20] = baseline_mean - 4.0 * np.sqrt(np.diag(baseline_cov))

        outliers = analyzer.detect_multivariate_outliers(test_data, baseline_mean, baseline_cov, alpha=0.05)

        assert isinstance(outliers, list)
        assert 10 in outliers or 20 in outliers  # Should detect at least one outlier

    @pytest.mark.unit
    def test_detect_multivariate_outliers_no_outliers(self, analyzer):
        """Test detecting outliers when none exist."""
        data, cov = generate_multivariate_in_control_data(n_samples=50, n_variables=5)

        baseline_mean = np.mean(data, axis=0)
        baseline_cov = np.cov(data, rowvar=False, ddof=1)

        outliers = analyzer.detect_multivariate_outliers(data, baseline_mean, baseline_cov, alpha=0.05)

        # Should have few or no outliers for in-control data
        assert isinstance(outliers, list)
        # May have some false positives, but should be relatively few
        assert len(outliers) < len(data) * 0.1  # Less than 10% false positives

    @pytest.mark.unit
    def test_hotelling_t2_high_dimensional(self, analyzer, config):
        """Test Hotelling T² with high-dimensional data."""
        data, cov = generate_high_dimensional_data(n_samples=150, n_variables=20)

        result = analyzer.create_hotelling_t2_chart(data, config=config)

        assert isinstance(result, MultivariateSPCResult)
        assert result.baseline_mean.shape == (20,)
        assert result.baseline_covariance.shape == (20, 20)
        assert len(result.hotelling_t2) == len(data)

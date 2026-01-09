"""
Unit tests for multivariate analysis.

Tests for MultivariateConfig, MultivariateResult, MultivariateAnalyzer, PCAAnalyzer, and ClusterAnalyzer.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from am_qadf.analytics.statistical_analysis.multivariate import (
    MultivariateConfig,
    MultivariateResult,
    MultivariateAnalyzer,
    PCAAnalyzer,
    ClusterAnalyzer,
)


class TestMultivariateConfig:
    """Test suite for MultivariateConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating MultivariateConfig with default values."""
        config = MultivariateConfig()

        assert config.pca_components is None
        assert config.pca_variance_threshold == 0.95
        assert config.clustering_method == "kmeans"
        assert config.n_clusters is None
        assert config.clustering_metric == "euclidean"
        assert config.scaling_method == "standard"
        assert config.confidence_level == 0.95
        assert config.significance_level == 0.05
        assert config.random_seed is None

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating MultivariateConfig with custom values."""
        config = MultivariateConfig(
            pca_components=5,
            pca_variance_threshold=0.99,
            clustering_method="dbscan",
            n_clusters=3,
            clustering_metric="manhattan",
            scaling_method="minmax",
            confidence_level=0.99,
            significance_level=0.01,
            random_seed=42,
        )

        assert config.pca_components == 5
        assert config.pca_variance_threshold == 0.99
        assert config.clustering_method == "dbscan"
        assert config.n_clusters == 3
        assert config.clustering_metric == "manhattan"
        assert config.scaling_method == "minmax"
        assert config.confidence_level == 0.99
        assert config.significance_level == 0.01
        assert config.random_seed == 42


class TestMultivariateResult:
    """Test suite for MultivariateResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating MultivariateResult."""
        result = MultivariateResult(
            success=True,
            method="PCA",
            feature_names=["feature1", "feature2"],
            analysis_results={"n_components": 2},
            explained_variance={"total_variance_explained": 0.95},
            component_loadings=pd.DataFrame({"PC1": [0.5, 0.5], "PC2": [0.5, -0.5]}),
            cluster_labels=None,
            analysis_time=1.5,
        )

        assert result.success is True
        assert result.method == "PCA"
        assert len(result.feature_names) == 2
        assert result.analysis_time == 1.5
        assert result.error_message is None

    @pytest.mark.unit
    def test_result_creation_with_error(self):
        """Test creating MultivariateResult with error."""
        result = MultivariateResult(
            success=False,
            method="PCA",
            feature_names=[],
            analysis_results={},
            explained_variance={},
            error_message="Test error",
        )

        assert result.success is False
        assert result.error_message == "Test error"


class TestMultivariateAnalyzer:
    """Test suite for MultivariateAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a MultivariateAnalyzer instance."""
        return MultivariateAnalyzer()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        return data

    @pytest.mark.unit
    def test_analyzer_creation_default(self):
        """Test creating MultivariateAnalyzer with default config."""
        analyzer = MultivariateAnalyzer()

        assert analyzer.config is not None
        assert isinstance(analyzer.config, MultivariateConfig)
        assert analyzer.analysis_cache == {}

    @pytest.mark.unit
    def test_analyzer_creation_custom_config(self):
        """Test creating MultivariateAnalyzer with custom config."""
        config = MultivariateConfig(pca_components=2)
        analyzer = MultivariateAnalyzer(config=config)

        assert analyzer.config.pca_components == 2

    @pytest.mark.unit
    def test_analyze_pca(self, analyzer, sample_data):
        """Test PCA analysis."""
        result = analyzer.analyze_pca(sample_data, n_components=2)

        assert isinstance(result, MultivariateResult)
        assert result.method == "PCA"
        assert result.success is True
        assert len(result.feature_names) == 3
        assert result.component_loadings is not None
        assert "total_variance_explained" in result.explained_variance

    @pytest.mark.unit
    def test_analyze_pca_auto_components(self, analyzer, sample_data):
        """Test PCA analysis with automatic component selection."""
        result = analyzer.analyze_pca(sample_data, n_components=None)

        assert isinstance(result, MultivariateResult)
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_pca_error_handling(self, analyzer):
        """Test error handling in PCA analysis."""
        empty_data = pd.DataFrame()

        result = analyzer.analyze_pca(empty_data)

        assert isinstance(result, MultivariateResult)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.unit
    def test_analyze_clustering_kmeans(self, analyzer, sample_data):
        """Test K-means clustering analysis."""
        result = analyzer.analyze_clustering(sample_data, method="kmeans", n_clusters=3)

        assert isinstance(result, MultivariateResult)
        assert result.method == "Clustering"
        assert result.success is True
        assert result.cluster_labels is not None
        assert len(result.cluster_labels) == len(sample_data)

    @pytest.mark.unit
    def test_analyze_clustering_dbscan(self, analyzer, sample_data):
        """Test DBSCAN clustering analysis."""
        result = analyzer.analyze_clustering(sample_data, method="dbscan")

        assert isinstance(result, MultivariateResult)
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_clustering_hierarchical(self, analyzer, sample_data):
        """Test hierarchical clustering analysis."""
        result = analyzer.analyze_clustering(sample_data, method="hierarchical", n_clusters=3)

        assert isinstance(result, MultivariateResult)
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_clustering_error_handling(self, analyzer):
        """Test error handling in clustering analysis."""
        empty_data = pd.DataFrame()

        result = analyzer.analyze_clustering(empty_data)

        assert isinstance(result, MultivariateResult)
        assert result.success is False

    @pytest.mark.unit
    def test_cache_result(self, analyzer):
        """Test caching analysis results."""
        result = MultivariateResult(
            success=True,
            method="test",
            feature_names=["feature1"],
            analysis_results={},
            explained_variance={},
        )

        analyzer._cache_result("test_method", result)

        cached = analyzer.get_cached_result("test_method", ["feature1"])
        assert cached is not None

    @pytest.mark.unit
    def test_get_cached_result_none(self, analyzer):
        """Test getting cached result when none exists."""
        cached = analyzer.get_cached_result("nonexistent", ["feature1"])
        assert cached is None

    @pytest.mark.unit
    def test_clear_cache(self, analyzer):
        """Test clearing analysis cache."""
        result = MultivariateResult(
            success=True,
            method="test",
            feature_names=["feature1"],
            analysis_results={},
            explained_variance={},
        )

        analyzer._cache_result("test_method", result)
        assert len(analyzer.analysis_cache) > 0

        analyzer.clear_cache()
        assert len(analyzer.analysis_cache) == 0


class TestPCAAnalyzer:
    """Test suite for PCAAnalyzer class."""

    @pytest.mark.unit
    def test_pca_analyzer_creation(self):
        """Test creating PCAAnalyzer."""
        analyzer = PCAAnalyzer()

        assert isinstance(analyzer, MultivariateAnalyzer)
        assert analyzer.method_name == "PCA"


class TestClusterAnalyzer:
    """Test suite for ClusterAnalyzer class."""

    @pytest.mark.unit
    def test_cluster_analyzer_creation(self):
        """Test creating ClusterAnalyzer."""
        analyzer = ClusterAnalyzer()

        assert isinstance(analyzer, MultivariateAnalyzer)
        assert analyzer.method_name == "Clustering"

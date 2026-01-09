"""
Unit tests for quality analysis.

Tests for QualityAnalysisConfig, QualityAnalysisResult, QualityAnalyzer, and QualityPredictor.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.analytics.process_analysis.quality_analysis import (
    QualityAnalysisConfig,
    QualityAnalysisResult,
    QualityAnalyzer,
    QualityPredictor,
)


class TestQualityAnalysisConfig:
    """Test suite for QualityAnalysisConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating QualityAnalysisConfig with default values."""
        config = QualityAnalysisConfig()

        assert config.model_type == "random_forest"
        assert config.test_size == 0.2
        assert config.quality_threshold == 0.8
        assert config.defect_threshold == 0.1
        assert config.confidence_level == 0.95
        assert config.random_seed is None

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating QualityAnalysisConfig with custom values."""
        config = QualityAnalysisConfig(
            model_type="gradient_boosting",
            test_size=0.3,
            quality_threshold=0.9,
            defect_threshold=0.05,
            random_seed=42,
        )

        assert config.model_type == "gradient_boosting"
        assert config.test_size == 0.3
        assert config.quality_threshold == 0.9
        assert config.defect_threshold == 0.05
        assert config.random_seed == 42


class TestQualityAnalysisResult:
    """Test suite for QualityAnalysisResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating QualityAnalysisResult."""
        result = QualityAnalysisResult(
            success=True,
            method="test_method",
            quality_metrics={"mean_quality": 0.8, "std_quality": 0.1},
            quality_predictions=np.array([0.7, 0.8, 0.9]),
            quality_classifications=np.array([0, 1, 1]),
            model_performance={"r2_score": 0.85, "mse": 0.05},
            analysis_time=1.5,
        )

        assert result.success is True
        assert result.method == "test_method"
        assert len(result.quality_metrics) == 2
        assert len(result.quality_predictions) == 3
        assert len(result.quality_classifications) == 3
        assert result.model_performance["r2_score"] == 0.85
        assert result.analysis_time == 1.5


class TestQualityAnalyzer:
    """Test suite for QualityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a QualityAnalyzer instance."""
        return QualityAnalyzer()

    @pytest.fixture
    def sample_process_data(self):
        """Create sample process data for testing."""
        np.random.seed(42)
        n_samples = 100
        data = pd.DataFrame(
            {
                "param1": np.random.randn(n_samples),
                "param2": np.random.randn(n_samples),
                "param3": np.random.randn(n_samples),
                "quality": np.random.rand(n_samples),  # Target variable
            }
        )
        return data

    @pytest.mark.unit
    def test_analyzer_creation_default(self):
        """Test creating QualityAnalyzer with default config."""
        analyzer = QualityAnalyzer()

        assert analyzer.config is not None
        assert analyzer.analysis_cache == {}

    @pytest.mark.unit
    def test_analyzer_creation_custom(self):
        """Test creating QualityAnalyzer with custom config."""
        config = QualityAnalysisConfig(model_type="gradient_boosting", random_seed=42)
        analyzer = QualityAnalyzer(config)

        assert analyzer.config.model_type == "gradient_boosting"
        assert analyzer.config.random_seed == 42

    @pytest.mark.unit
    def test_analyze_quality_prediction_random_forest(self, analyzer, sample_process_data):
        """Test quality prediction with random forest."""
        result = analyzer.analyze_quality_prediction(
            sample_process_data,
            quality_target="quality",
            feature_names=["param1", "param2", "param3"],
        )

        assert isinstance(result, QualityAnalysisResult)
        assert result.success is True
        assert result.method == "QualityPrediction"
        assert len(result.quality_predictions) > 0
        assert len(result.quality_classifications) > 0
        assert "r2_score" in result.model_performance

    @pytest.mark.unit
    def test_analyze_quality_prediction_gradient_boosting(self, analyzer, sample_process_data):
        """Test quality prediction with gradient boosting."""
        config = QualityAnalysisConfig(model_type="gradient_boosting", random_seed=42)
        analyzer = QualityAnalyzer(config)

        result = analyzer.analyze_quality_prediction(
            sample_process_data,
            quality_target="quality",
            feature_names=["param1", "param2", "param3"],
        )

        assert isinstance(result, QualityAnalysisResult)
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_quality_control(self, analyzer, sample_process_data):
        """Test quality control analysis."""
        result = analyzer.analyze_quality_control(
            sample_process_data,
            quality_target="quality",
            feature_names=["param1", "param2", "param3"],
        )

        assert isinstance(result, QualityAnalysisResult)
        assert result.success is True
        assert "quality_metrics" in result.quality_metrics or len(result.quality_metrics) > 0


class TestQualityPredictor:
    """Test suite for QualityPredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create a QualityPredictor instance."""
        return QualityPredictor()

    @pytest.fixture
    def sample_process_data(self):
        """Create sample process data for testing."""
        np.random.seed(42)
        n_samples = 100
        data = pd.DataFrame(
            {
                "param1": np.random.randn(n_samples),
                "param2": np.random.randn(n_samples),
                "quality": np.random.rand(n_samples),
            }
        )
        return data

    @pytest.mark.unit
    def test_predictor_creation(self, predictor):
        """Test creating QualityPredictor."""
        assert predictor is not None
        assert predictor.config is not None

    @pytest.mark.unit
    def test_predict_quality(self, predictor, sample_process_data):
        """Test predicting quality."""
        # Train model first
        predictor.analyze_quality_prediction(
            sample_process_data,
            quality_target="quality",
            feature_names=["param1", "param2"],
        )

        # Make predictions
        new_data = pd.DataFrame({"param1": [0.5, 1.0, -0.5], "param2": [0.3, -0.2, 0.8]})

        predictions = predictor.predict_quality(new_data, feature_names=["param1", "param2"])

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3

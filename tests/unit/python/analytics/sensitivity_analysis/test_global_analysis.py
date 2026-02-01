"""
Unit tests for global sensitivity analysis.

Tests for SensitivityConfig, SensitivityResult, GlobalSensitivityAnalyzer, SobolAnalyzer, and MorrisAnalyzer.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from am_qadf.analytics.sensitivity_analysis.global_analysis import (
    SensitivityConfig,
    SensitivityResult,
    GlobalSensitivityAnalyzer,
    SobolAnalyzer,
    MorrisAnalyzer,
)


class TestSensitivityConfig:
    """Test suite for SensitivityConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating SensitivityConfig with default values."""
        config = SensitivityConfig()

        assert config.sample_size == 1000
        assert config.confidence_level == 0.95
        assert config.random_seed is None
        assert config.sobol_order == 2
        assert config.sobol_n_bootstrap == 100
        assert config.morris_levels == 10
        assert config.morris_num_trajectories == 10
        assert config.parallel_processing is True
        assert config.max_workers == 4
        assert config.memory_limit_gb == 8.0

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating SensitivityConfig with custom values."""
        config = SensitivityConfig(
            sample_size=500,
            confidence_level=0.99,
            random_seed=42,
            sobol_order=3,
            sobol_n_bootstrap=200,
            morris_levels=20,
            morris_num_trajectories=20,
            parallel_processing=False,
            max_workers=8,
            memory_limit_gb=16.0,
        )

        assert config.sample_size == 500
        assert config.confidence_level == 0.99
        assert config.random_seed == 42
        assert config.sobol_order == 3
        assert config.sobol_n_bootstrap == 200
        assert config.morris_levels == 20
        assert config.morris_num_trajectories == 20
        assert config.parallel_processing is False
        assert config.max_workers == 8
        assert config.memory_limit_gb == 16.0


class TestSensitivityResult:
    """Test suite for SensitivityResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating SensitivityResult."""
        result = SensitivityResult(
            success=True,
            method="Sobol",
            parameter_names=["param1", "param2"],
            sensitivity_indices={"S1_param1": 0.5, "S1_param2": 0.3},
            confidence_intervals={"S1_param1": (0.4, 0.6)},
            analysis_time=1.5,
            sample_size=1000,
        )

        assert result.success is True
        assert result.method == "Sobol"
        assert len(result.parameter_names) == 2
        assert len(result.sensitivity_indices) == 2
        assert result.analysis_time == 1.5
        assert result.sample_size == 1000
        assert result.error_message is None

    @pytest.mark.unit
    def test_result_creation_with_error(self):
        """Test creating SensitivityResult with error."""
        result = SensitivityResult(
            success=False,
            method="Sobol",
            parameter_names=[],
            sensitivity_indices={},
            confidence_intervals={},
            analysis_time=0.0,
            sample_size=0,
            error_message="Test error",
        )

        assert result.success is False
        assert result.error_message == "Test error"


class TestGlobalSensitivityAnalyzer:
    """Test suite for GlobalSensitivityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a GlobalSensitivityAnalyzer instance."""
        return GlobalSensitivityAnalyzer()

    @pytest.fixture
    def simple_model_function(self):
        """Create a simple model function for testing."""

        def model(params):
            # Simple linear model: y = 2*x1 + x2
            if isinstance(params, np.ndarray):
                return 2 * params[0] + params[1]
            return 2 * params[0] + params[1]

        return model

    @pytest.fixture
    def parameter_bounds(self):
        """Create parameter bounds for testing."""
        return {"param1": (0.0, 10.0), "param2": (0.0, 10.0)}

    @pytest.mark.unit
    def test_analyzer_creation_default(self):
        """Test creating GlobalSensitivityAnalyzer with default config."""
        analyzer = GlobalSensitivityAnalyzer()

        assert analyzer.config is not None
        assert isinstance(analyzer.config, SensitivityConfig)
        assert analyzer.analysis_cache == {}

    @pytest.mark.unit
    def test_analyzer_creation_custom_config(self):
        """Test creating GlobalSensitivityAnalyzer with custom config."""
        config = SensitivityConfig(sample_size=500)
        analyzer = GlobalSensitivityAnalyzer(config=config)

        assert analyzer.config.sample_size == 500

    @pytest.mark.unit
    @patch("am_qadf.analytics.sensitivity_analysis.global_analysis.SALIB_AVAILABLE", True)
    def test_analyze_sobol(self, analyzer, simple_model_function, parameter_bounds):
        """Test Sobol sensitivity analysis."""
        with (
            patch("am_qadf.analytics.sensitivity_analysis.global_analysis.saltelli.sample") as mock_sample,
            patch("am_qadf.analytics.sensitivity_analysis.global_analysis.sobol.analyze") as mock_analyze,
        ):

            # Mock sample generation
            mock_sample.return_value = np.random.uniform(0, 1, (100, 2))

            # Mock Sobol analysis results
            mock_analyze.return_value = {
                "S1": np.array([0.8, 0.2]),
                "ST": np.array([0.9, 0.3]),
                "S1_conf": np.array([[0.7, 0.9], [0.1, 0.3]]),
                "ST_conf": np.array([[0.8, 1.0], [0.2, 0.4]]),
            }

            result = analyzer.analyze_sobol(
                simple_model_function,
                parameter_bounds,
                parameter_names=["param1", "param2"],
            )

            assert isinstance(result, SensitivityResult)
            assert result.method == "Sobol"
            assert result.success is True
            assert len(result.parameter_names) == 2

    @pytest.mark.unit
    @patch("am_qadf.analytics.sensitivity_analysis.global_analysis.SALIB_AVAILABLE", False)
    def test_analyze_sobol_fallback(self, analyzer, simple_model_function, parameter_bounds):
        """Test Sobol analysis fallback when SALib is not available."""
        result = analyzer.analyze_sobol(
            simple_model_function,
            parameter_bounds,
            parameter_names=["param1", "param2"],
        )

        assert isinstance(result, SensitivityResult)
        assert result.method == "Sobol"
        # May succeed or fail depending on fallback implementation

    @pytest.mark.unit
    @patch("am_qadf.analytics.sensitivity_analysis.global_analysis.SALIB_AVAILABLE", True)
    def test_analyze_morris(self, analyzer, simple_model_function, parameter_bounds):
        """Test Morris sensitivity analysis."""
        with (
            patch("am_qadf.analytics.sensitivity_analysis.global_analysis.morris.sample") as mock_sample,
            patch("am_qadf.analytics.sensitivity_analysis.global_analysis.morris_analyze.analyze") as mock_analyze,
        ):

            # Mock sample generation
            mock_sample.return_value = np.random.uniform(0, 1, (50, 2))

            # Mock Morris analysis results
            mock_analyze.return_value = {
                "mu": np.array([1.5, 0.5]),
                "sigma": np.array([0.2, 0.1]),
                "mu_star": np.array([1.6, 0.6]),
                "mu_conf": np.array([[1.4, 1.6], [0.4, 0.6]]),
                "sigma_conf": np.array([[0.1, 0.3], [0.05, 0.15]]),
                "mu_star_conf": np.array([[1.5, 1.7], [0.5, 0.7]]),
            }

            result = analyzer.analyze_morris(
                simple_model_function,
                parameter_bounds,
                parameter_names=["param1", "param2"],
            )

            assert isinstance(result, SensitivityResult)
            assert result.method == "Morris"
            assert result.success is True
            assert len(result.parameter_names) == 2

    @pytest.mark.unit
    @patch("am_qadf.analytics.sensitivity_analysis.global_analysis.SALIB_AVAILABLE", False)
    def test_analyze_morris_fallback(self, analyzer, simple_model_function, parameter_bounds):
        """Test Morris analysis fallback when SALib is not available."""
        result = analyzer.analyze_morris(
            simple_model_function,
            parameter_bounds,
            parameter_names=["param1", "param2"],
        )

        assert isinstance(result, SensitivityResult)
        assert result.method == "Morris"

    @pytest.mark.unit
    def test_analyze_variance_based(self, analyzer, simple_model_function, parameter_bounds):
        """Test variance-based sensitivity analysis."""
        analyzer.config.sample_size = 100  # Use smaller sample for testing

        result = analyzer.analyze_variance_based(
            simple_model_function,
            parameter_bounds,
            parameter_names=["param1", "param2"],
        )

        assert isinstance(result, SensitivityResult)
        assert result.method == "VarianceBased"
        assert result.success is True
        assert len(result.parameter_names) == 2
        assert len(result.sensitivity_indices) > 0

    @pytest.mark.unit
    def test_analyze_sobol_error_handling(self, analyzer, parameter_bounds):
        """Test error handling in Sobol analysis."""

        def failing_model(params):
            raise ValueError("Model error")

        result = analyzer.analyze_sobol(failing_model, parameter_bounds, parameter_names=["param1", "param2"])

        assert isinstance(result, SensitivityResult)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.unit
    def test_analyze_morris_error_handling(self, analyzer, parameter_bounds):
        """Test error handling in Morris analysis."""

        def failing_model(params):
            raise ValueError("Model error")

        result = analyzer.analyze_morris(failing_model, parameter_bounds, parameter_names=["param1", "param2"])

        assert isinstance(result, SensitivityResult)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.unit
    def test_evaluate_model_parallel(self, analyzer, simple_model_function):
        """Test parallel model evaluation."""
        analyzer.config.parallel_processing = False  # Disable for testing
        param_values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        results = analyzer._evaluate_model_parallel(simple_model_function, param_values)

        assert len(results) == 3
        assert results[0] == 2 * 1.0 + 2.0  # 4.0
        assert results[1] == 2 * 3.0 + 4.0  # 10.0
        assert results[2] == 2 * 5.0 + 6.0  # 16.0


class TestSobolAnalyzer:
    """Test suite for SobolAnalyzer class."""

    @pytest.mark.unit
    def test_sobol_analyzer_creation(self):
        """Test creating SobolAnalyzer."""
        analyzer = SobolAnalyzer()

        assert isinstance(analyzer, GlobalSensitivityAnalyzer)
        assert analyzer.config is not None


class TestMorrisAnalyzer:
    """Test suite for MorrisAnalyzer class."""

    @pytest.mark.unit
    def test_morris_analyzer_creation(self):
        """Test creating MorrisAnalyzer."""
        analyzer = MorrisAnalyzer()

        assert isinstance(analyzer, GlobalSensitivityAnalyzer)
        assert analyzer.config is not None

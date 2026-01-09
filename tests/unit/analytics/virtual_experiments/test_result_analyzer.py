"""
Unit tests for result analyzer.

Tests for AnalysisResult and VirtualExperimentResultAnalyzer classes.
"""

import pytest
import numpy as np
from am_qadf.analytics.virtual_experiments.result_analyzer import (
    AnalysisResult,
    VirtualExperimentResultAnalyzer,
)


class TestAnalysisResult:
    """Test suite for AnalysisResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating AnalysisResult."""
        result = AnalysisResult(
            experiment_id="exp1",
            analysis_type="comprehensive",
            parameter_names=["power", "velocity"],
            response_names=["quality", "density"],
            response_statistics={
                "quality": {"mean": 0.8, "std": 0.1},
                "density": {"mean": 0.95, "std": 0.05},
            },
            correlations={"quality": {"power": 0.6, "velocity": 0.4}},
            parameter_interactions={"power_velocity": 0.3},
            analysis_time=1.5,
            sample_size=100,
            success=True,
        )

        assert result.experiment_id == "exp1"
        assert result.analysis_type == "comprehensive"
        assert len(result.parameter_names) == 2
        assert len(result.response_names) == 2
        assert result.response_statistics["quality"]["mean"] == 0.8
        assert result.analysis_time == 1.5
        assert result.sample_size == 100
        assert result.success is True


class TestVirtualExperimentResultAnalyzer:
    """Test suite for VirtualExperimentResultAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a VirtualExperimentResultAnalyzer instance."""
        return VirtualExperimentResultAnalyzer()

    @pytest.fixture
    def sample_experiment_results(self):
        """Create sample experiment results for testing."""
        return [
            {
                "input_parameters": {"power": 200.0, "velocity": 1000.0},
                "output_responses": {"quality": 0.8, "density": 0.95},
            },
            {
                "input_parameters": {"power": 250.0, "velocity": 1200.0},
                "output_responses": {"quality": 0.85, "density": 0.96},
            },
            {
                "input_parameters": {"power": 300.0, "velocity": 1500.0},
                "output_responses": {"quality": 0.9, "density": 0.97},
            },
        ]

    @pytest.mark.unit
    def test_analyzer_creation(self, analyzer):
        """Test creating VirtualExperimentResultAnalyzer."""
        assert analyzer is not None
        assert analyzer.analysis_cache == {}

    @pytest.mark.unit
    def test_analyze_results(self, analyzer, sample_experiment_results):
        """Test analyzing experiment results."""
        result = analyzer.analyze_results(
            sample_experiment_results,
            parameter_names=["power", "velocity"],
            response_names=["quality", "density"],
        )

        assert isinstance(result, AnalysisResult)
        assert result.success is True
        assert len(result.parameter_names) == 2
        assert len(result.response_names) == 2
        assert result.sample_size == 3
        assert len(result.response_statistics) > 0
        assert len(result.correlations) > 0

    @pytest.mark.unit
    def test_analyze_results_with_parameters_key(self, analyzer):
        """Test analyzing results with 'parameters' key instead of 'input_parameters'."""
        results = [
            {
                "parameters": {"power": 200.0, "velocity": 1000.0},
                "responses": {"quality": 0.8},
            }
        ]

        result = analyzer.analyze_results(results, parameter_names=["power", "velocity"], response_names=["quality"])

        assert isinstance(result, AnalysisResult)
        assert result.success is True

    @pytest.mark.unit
    def test_analyze_results_empty(self, analyzer):
        """Test analyzing empty results."""
        result = analyzer.analyze_results([], parameter_names=["power"], response_names=["quality"])

        assert isinstance(result, AnalysisResult)
        assert result.success is False or result.sample_size == 0

    @pytest.mark.unit
    def test_analyze_results_error_handling(self, analyzer):
        """Test error handling in analyze_results."""
        # Pass invalid data that might cause errors
        result = analyzer.analyze_results([{"invalid": "data"}], parameter_names=["power"], response_names=["quality"])

        assert isinstance(result, AnalysisResult)
        # Should handle error gracefully
        assert result.success is False or result.sample_size == 0

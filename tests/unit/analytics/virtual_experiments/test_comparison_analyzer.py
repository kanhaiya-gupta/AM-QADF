"""
Unit tests for comparison analyzer.

Tests for ComparisonResult and ComparisonAnalyzer classes.
"""

import pytest
import numpy as np
from am_qadf.analytics.virtual_experiments.comparison_analyzer import (
    ComparisonResult,
    ComparisonAnalyzer,
)


class TestComparisonResult:
    """Test suite for ComparisonResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating ComparisonResult."""
        result = ComparisonResult(
            success=True,
            parameter_rankings_virtual={"power": 0.8, "velocity": 0.6},
            parameter_rankings_sensitivity={"power": 0.75, "velocity": 0.65},
            ranking_correlation=0.95,
            agreement_metrics={"top3_agreement": 1.0, "overall_agreement": 0.9},
            discrepancies=["temperature"],
        )

        assert result.success is True
        assert len(result.parameter_rankings_virtual) == 2
        assert len(result.parameter_rankings_sensitivity) == 2
        assert result.ranking_correlation == 0.95
        assert result.agreement_metrics["top3_agreement"] == 1.0
        assert len(result.discrepancies) == 1


class TestComparisonAnalyzer:
    """Test suite for ComparisonAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a ComparisonAnalyzer instance."""
        return ComparisonAnalyzer()

    @pytest.fixture
    def sample_virtual_results(self):
        """Create sample virtual experiment results."""
        return {
            "correlations": {
                "quality": {
                    "power": {"pearson": 0.8},
                    "velocity": {"pearson": 0.6},
                    "temperature": {"pearson": 0.4},
                }
            }
        }

    @pytest.fixture
    def sample_sensitivity_results(self):
        """Create sample sensitivity analysis results."""
        return {
            "sensitivity_indices": {
                "S1_power": 0.75,
                "S1_velocity": 0.65,
                "S1_temperature": 0.5,
            }
        }

    @pytest.mark.unit
    def test_analyzer_creation(self, analyzer):
        """Test creating ComparisonAnalyzer."""
        assert analyzer is not None

    @pytest.mark.unit
    def test_compare_parameter_importance(self, analyzer, sample_virtual_results, sample_sensitivity_results):
        """Test comparing parameter importance."""
        result = analyzer.compare_parameter_importance(
            sample_virtual_results, sample_sensitivity_results, response_name="quality"
        )

        assert isinstance(result, ComparisonResult)
        assert result.success is True
        assert len(result.parameter_rankings_virtual) > 0
        assert len(result.parameter_rankings_sensitivity) > 0
        assert -1.0 <= result.ranking_correlation <= 1.0
        assert len(result.agreement_metrics) > 0

    @pytest.mark.unit
    def test_compare_parameter_importance_different_structure(self, analyzer):
        """Test comparing with different result structures."""
        virtual_results = {"correlations": {"quality": {"power": 0.8, "velocity": 0.6}}}  # Direct float value

        sensitivity_results = {"sensitivity_indices": {"mu_star_power": 0.75, "mu_star_velocity": 0.65}}

        result = analyzer.compare_parameter_importance(virtual_results, sensitivity_results, response_name="quality")

        assert isinstance(result, ComparisonResult)
        assert result.success is True

    @pytest.mark.unit
    def test_compare_parameter_importance_error_handling(self, analyzer):
        """Test error handling in comparison."""
        # Pass invalid data
        result = analyzer.compare_parameter_importance({}, {}, response_name="quality")

        assert isinstance(result, ComparisonResult)
        # Should handle error gracefully
        assert result.success is False or len(result.parameter_rankings_virtual) == 0

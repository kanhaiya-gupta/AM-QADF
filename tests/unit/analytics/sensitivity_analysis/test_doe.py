"""
Unit tests for design of experiments.

Tests for DOEConfig, ExperimentalDesign, ExperimentalDesigner, FactorialDesign, and ResponseSurfaceDesign.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from am_qadf.analytics.sensitivity_analysis.doe import (
    DOEConfig,
    ExperimentalDesign,
    ExperimentalDesigner,
    FactorialDesign,
    ResponseSurfaceDesign,
)


class TestDOEConfig:
    """Test suite for DOEConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating DOEConfig with default values."""
        config = DOEConfig()

        assert config.design_type == "factorial"
        assert config.randomization is True
        assert config.blocking is False
        assert config.replication == 1
        assert config.factorial_levels == 2
        assert config.center_points == 0
        assert config.rs_design_type == "ccd"
        assert config.rs_alpha == 1.414
        assert config.optimal_criterion == "d_optimal"
        assert config.optimal_iterations == 1000
        assert config.confidence_level == 0.95
        assert config.significance_level == 0.05

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating DOEConfig with custom values."""
        config = DOEConfig(
            design_type="response_surface",
            randomization=False,
            blocking=True,
            replication=3,
            factorial_levels=3,
            center_points=5,
            rs_design_type="bbd",
            rs_alpha=2.0,
            optimal_criterion="a_optimal",
            optimal_iterations=2000,
            confidence_level=0.99,
            significance_level=0.01,
        )

        assert config.design_type == "response_surface"
        assert config.randomization is False
        assert config.blocking is True
        assert config.replication == 3
        assert config.factorial_levels == 3
        assert config.center_points == 5
        assert config.rs_design_type == "bbd"
        assert config.rs_alpha == 2.0
        assert config.optimal_criterion == "a_optimal"
        assert config.optimal_iterations == 2000
        assert config.confidence_level == 0.99
        assert config.significance_level == 0.01


class TestExperimentalDesign:
    """Test suite for ExperimentalDesign dataclass."""

    @pytest.mark.unit
    def test_design_creation(self):
        """Test creating ExperimentalDesign."""
        from datetime import datetime

        design_matrix = pd.DataFrame({"param1": [0, 1, 0, 1], "param2": [0, 0, 1, 1]})
        parameter_bounds = {"param1": (0.0, 10.0), "param2": (0.0, 10.0)}

        design = ExperimentalDesign(
            design_type="2^k_factorial",
            design_matrix=design_matrix,
            parameter_names=["param1", "param2"],
            parameter_bounds=parameter_bounds,
            design_points=4,
            design_quality={"d_efficiency": 0.9},
            randomization_seed=42,
            creation_time=datetime.now(),
        )

        assert design.design_type == "2^k_factorial"
        assert len(design.design_matrix) == 4
        assert len(design.parameter_names) == 2
        assert design.design_points == 4
        assert "d_efficiency" in design.design_quality


class TestExperimentalDesigner:
    """Test suite for ExperimentalDesigner class."""

    @pytest.fixture
    def designer(self):
        """Create an ExperimentalDesigner instance."""
        return ExperimentalDesigner()

    @pytest.fixture
    def parameter_bounds(self):
        """Create parameter bounds for testing."""
        return {"param1": (0.0, 10.0), "param2": (0.0, 10.0)}

    @pytest.mark.unit
    def test_designer_creation_default(self):
        """Test creating ExperimentalDesigner with default config."""
        designer = ExperimentalDesigner()

        assert designer.config is not None
        assert isinstance(designer.config, DOEConfig)
        assert designer.design_cache == {}

    @pytest.mark.unit
    def test_designer_creation_custom_config(self):
        """Test creating ExperimentalDesigner with custom config."""
        config = DOEConfig(factorial_levels=3)
        designer = ExperimentalDesigner(config=config)

        assert designer.config.factorial_levels == 3

    @pytest.mark.unit
    def test_create_factorial_design_2k(self, designer, parameter_bounds):
        """Test creating 2^k factorial design."""
        design = designer.create_factorial_design(parameter_bounds, parameter_names=["param1", "param2"], levels=2)

        assert isinstance(design, ExperimentalDesign)
        assert design.design_type == "2^k_factorial"
        assert len(design.design_matrix) >= 4  # At least 2^2 = 4 points
        assert len(design.parameter_names) == 2

    @pytest.mark.unit
    def test_create_factorial_design_3k(self, designer, parameter_bounds):
        """Test creating 3^k factorial design."""
        design = designer.create_factorial_design(parameter_bounds, parameter_names=["param1", "param2"], levels=3)

        assert isinstance(design, ExperimentalDesign)
        assert design.design_type == "3^k_factorial"
        assert len(design.design_matrix) >= 9  # At least 3^2 = 9 points

    @pytest.mark.unit
    def test_create_factorial_design_with_center_points(self, designer, parameter_bounds):
        """Test creating factorial design with center points."""
        designer.config.center_points = 3
        design = designer.create_factorial_design(parameter_bounds, parameter_names=["param1", "param2"], levels=2)

        assert isinstance(design, ExperimentalDesign)
        # Should have at least 4 factorial points + 3 center points
        assert len(design.design_matrix) >= 7

    @pytest.mark.unit
    def test_create_factorial_design_invalid_levels(self, designer, parameter_bounds):
        """Test creating factorial design with invalid levels."""
        with pytest.raises(ValueError, match="Unsupported number of levels"):
            designer.create_factorial_design(parameter_bounds, parameter_names=["param1", "param2"], levels=4)

    @pytest.mark.unit
    def test_create_response_surface_design_ccd(self, designer, parameter_bounds):
        """Test creating central composite design."""
        design = designer.create_response_surface_design(
            parameter_bounds, parameter_names=["param1", "param2"], design_type="ccd"
        )

        assert isinstance(design, ExperimentalDesign)
        assert "ccd" in design.design_type.lower() or "response_surface" in design.design_type.lower()
        assert len(design.parameter_names) == 2

    @pytest.mark.unit
    def test_create_response_surface_design_bbd(self, designer, parameter_bounds):
        """Test creating Box-Behnken design."""
        # Box-Behnken requires at least 3 factors
        # Create bounds with 3 parameters
        bbd_bounds = {"param1": (0, 1), "param2": (0, 1), "param3": (0, 1)}
        design = designer.create_response_surface_design(
            bbd_bounds,
            parameter_names=["param1", "param2", "param3"],
            design_type="bbd",
        )

        assert isinstance(design, ExperimentalDesign)
        assert len(design.parameter_names) == 3

    @pytest.mark.unit
    def test_create_response_surface_design_d_optimal(self, designer, parameter_bounds):
        """Test creating D-optimal design."""
        design = designer.create_response_surface_design(
            parameter_bounds,
            parameter_names=["param1", "param2"],
            design_type="d_optimal",
        )

        assert isinstance(design, ExperimentalDesign)
        assert len(design.parameter_names) == 2

    @pytest.mark.unit
    def test_create_response_surface_design_invalid_type(self, designer, parameter_bounds):
        """Test creating response surface design with invalid type."""
        with pytest.raises(ValueError, match="Unsupported response surface design type"):
            designer.create_response_surface_design(
                parameter_bounds,
                parameter_names=["param1", "param2"],
                design_type="invalid",
            )

    @pytest.mark.unit
    def test_create_optimal_design(self, designer, parameter_bounds):
        """Test creating optimal experimental design."""
        design = designer.create_optimal_design(parameter_bounds, parameter_names=["param1", "param2"], n_points=10)

        assert isinstance(design, ExperimentalDesign)
        assert len(design.parameter_names) == 2
        assert design.design_points >= 10

    @pytest.mark.unit
    def test_cache_design(self, designer, parameter_bounds):
        """Test caching experimental designs."""
        design = designer.create_factorial_design(parameter_bounds, parameter_names=["param1", "param2"], levels=2)

        assert len(designer.design_cache) > 0

    @pytest.mark.unit
    def test_get_cached_design(self, designer, parameter_bounds):
        """Test getting cached experimental design."""
        design = designer.create_factorial_design(parameter_bounds, parameter_names=["param1", "param2"], levels=2)

        cached = designer.get_cached_design("factorial", ["param1", "param2"])
        assert cached is not None

    @pytest.mark.unit
    def test_get_cached_design_none(self, designer):
        """Test getting cached design when none exists."""
        cached = designer.get_cached_design("nonexistent", ["param1"])
        assert cached is None

    @pytest.mark.unit
    def test_clear_design_cache(self, designer, parameter_bounds):
        """Test clearing design cache."""
        designer.create_factorial_design(parameter_bounds, parameter_names=["param1", "param2"], levels=2)

        assert len(designer.design_cache) > 0
        designer.clear_design_cache()
        assert len(designer.design_cache) == 0


class TestFactorialDesign:
    """Test suite for FactorialDesign class."""

    @pytest.mark.unit
    def test_factorial_design_creation(self):
        """Test creating FactorialDesign."""
        designer = FactorialDesign()

        assert isinstance(designer, ExperimentalDesigner)
        assert designer.method_name == "Factorial"


class TestResponseSurfaceDesign:
    """Test suite for ResponseSurfaceDesign class."""

    @pytest.mark.unit
    def test_response_surface_design_creation(self):
        """Test creating ResponseSurfaceDesign."""
        designer = ResponseSurfaceDesign()

        assert isinstance(designer, ExperimentalDesigner)
        assert designer.method_name == "ResponseSurface"

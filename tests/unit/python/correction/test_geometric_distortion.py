"""
Unit tests for geometric distortion (C++ wrapper).

Tests DistortionModel, ScalingModel, RotationModel, WarpingModel, CombinedDistortionModel.
Models wrap C++ GeometricCorrection; apply() is not fully implemented (NotImplementedError)
except for CombinedDistortionModel which chains sub-models.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from am_qadf.correction.geometric_distortion import (
    DistortionModel,
    ScalingModel,
    RotationModel,
    WarpingModel,
    CombinedDistortionModel,
)


class TestDistortionModel:
    """Test suite for DistortionModel abstract base class."""

    @pytest.mark.unit
    def test_distortion_model_abstract(self):
        """Test that DistortionModel cannot be instantiated."""
        with pytest.raises(TypeError):
            DistortionModel()


class TestScalingModel:
    """Test suite for ScalingModel (C++ wrapper)."""

    @pytest.mark.unit
    def test_scaling_model_creation(self):
        """Test creating ScalingModel with scale_factors tuple."""
        model = ScalingModel(scale_factors=(2.0, 3.0, 4.0))
        assert model.scale_factors == (2.0, 3.0, 4.0)

    @pytest.mark.unit
    def test_scaling_model_apply_not_implemented(self):
        """Test that ScalingModel.apply(grid) raises NotImplementedError."""
        model = ScalingModel(scale_factors=(1.0, 1.0, 1.0))
        mock_grid = Mock()
        with pytest.raises(NotImplementedError, match="ScalingModel.apply|not yet fully implemented"):
            model.apply(mock_grid)


class TestRotationModel:
    """Test suite for RotationModel (C++ wrapper)."""

    @pytest.mark.unit
    def test_rotation_model_creation(self):
        """Test creating RotationModel with rotation_angles tuple."""
        model = RotationModel(rotation_angles=(0.0, 0.0, 1.57))
        assert model.rotation_angles == (0.0, 0.0, 1.57)

    @pytest.mark.unit
    def test_rotation_model_apply_not_implemented(self):
        """Test that RotationModel.apply(grid) raises NotImplementedError."""
        model = RotationModel(rotation_angles=(0.0, 0.0, 0.0))
        mock_grid = Mock()
        with pytest.raises(NotImplementedError, match="RotationModel.apply|not yet fully implemented"):
            model.apply(mock_grid)


class TestWarpingModel:
    """Test suite for WarpingModel (C++ wrapper)."""

    @pytest.mark.unit
    def test_warping_model_creation(self):
        """Test creating WarpingModel with distortion_map."""
        model = WarpingModel(distortion_map={"type": "lens", "k1": 0.1})
        assert model.distortion_map == {"type": "lens", "k1": 0.1}

    @pytest.mark.unit
    def test_warping_model_apply_not_implemented(self):
        """Test that WarpingModel.apply(grid) raises NotImplementedError."""
        model = WarpingModel(distortion_map={})
        mock_grid = Mock()
        with pytest.raises(NotImplementedError, match="WarpingModel.apply|not yet fully implemented"):
            model.apply(mock_grid)


class TestCombinedDistortionModel:
    """Test suite for CombinedDistortionModel."""

    @pytest.mark.unit
    def test_combined_model_creation(self):
        """Test creating CombinedDistortionModel with list of models."""
        scaling = ScalingModel(scale_factors=(1.0, 1.0, 1.0))
        rotation = RotationModel(rotation_angles=(0.0, 0.0, 0.0))
        model = CombinedDistortionModel([scaling, rotation])
        assert len(model.models) == 2
        assert model.models[0] is scaling
        assert model.models[1] is rotation

    @pytest.mark.unit
    def test_combined_model_apply_empty_returns_grid(self):
        """Test CombinedDistortionModel with no models returns grid unchanged."""
        model = CombinedDistortionModel([])
        mock_grid = Mock()
        result = model.apply(mock_grid)
        assert result is mock_grid

    @pytest.mark.unit
    def test_combined_model_apply_chains_models(self):
        """Test CombinedDistortionModel.apply chains model.apply (first model raises)."""
        scaling = ScalingModel(scale_factors=(1.0, 1.0, 1.0))
        model = CombinedDistortionModel([scaling])
        mock_grid = Mock()
        with pytest.raises(NotImplementedError):
            model.apply(mock_grid)

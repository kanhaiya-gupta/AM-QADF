"""
Unit tests for validation (C++ wrapper).

Tests AlignmentQuality, ValidationMetrics, and CorrectionValidator.
CorrectionValidator delegates to am_qadf_native.correction.Validation; skip if not built.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from am_qadf.correction.validation import (
    AlignmentQuality,
    ValidationMetrics,
    CorrectionValidator,
)


class TestAlignmentQuality:
    """Test suite for AlignmentQuality dataclass."""

    @pytest.mark.unit
    def test_alignment_quality_creation(self):
        """Test creating AlignmentQuality."""
        q = AlignmentQuality(
            spatial_error=0.1,
            temporal_error=0.05,
            overall_quality=0.9,
            metrics={"rmse": 0.1},
        )
        assert q.spatial_error == 0.1
        assert q.temporal_error == 0.05
        assert q.overall_quality == 0.9
        assert q.metrics == {"rmse": 0.1}

    @pytest.mark.unit
    def test_alignment_quality_metrics_default(self):
        """Test AlignmentQuality with default metrics."""
        q = AlignmentQuality(spatial_error=0.0, temporal_error=0.0, overall_quality=1.0)
        assert q.metrics == {}


class TestValidationMetrics:
    """Test suite for ValidationMetrics dataclass."""

    @pytest.mark.unit
    def test_validation_metrics_creation(self):
        """Test creating ValidationMetrics."""
        m = ValidationMetrics(
            is_valid=True,
            errors=[],
            warnings=["minor"],
            metrics={"max": 1.0},
        )
        assert m.is_valid is True
        assert m.errors == []
        assert m.warnings == ["minor"]
        assert m.metrics == {"max": 1.0}

    @pytest.mark.unit
    def test_validation_metrics_defaults(self):
        """Test ValidationMetrics with default lists."""
        m = ValidationMetrics(is_valid=False)
        assert m.errors == []
        assert m.warnings == []
        assert m.metrics == {}


class TestCorrectionValidator:
    """Test suite for CorrectionValidator (C++ wrapper)."""

    @pytest.mark.unit
    def test_correction_validator_requires_cpp(self):
        """Test that CorrectionValidator raises when am_qadf_native is not available."""
        import am_qadf.correction.validation as val_mod
        with patch.object(val_mod, "CPP_AVAILABLE", False):
            with pytest.raises(ImportError, match=r"C\+\+ bindings not available"):
                CorrectionValidator()

    @pytest.mark.unit
    def test_correction_validator_creation(self):
        """Test creating CorrectionValidator when C++ is available."""
        pytest.importorskip("am_qadf_native.correction", reason="Validation C++ bindings required")
        v = CorrectionValidator()
        assert v._validator is not None

    @pytest.mark.unit
    def test_validate_signal_data(self):
        """Test validate_signal_data returns ValidationMetrics."""
        pytest.importorskip("am_qadf_native.correction", reason="Validation C++ bindings required")
        v = CorrectionValidator()
        values = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        result = v.validate_signal_data(values, min_value=0.0, max_value=1.0)
        assert isinstance(result, ValidationMetrics)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.metrics, dict)

    @pytest.mark.unit
    def test_validate_coordinates(self):
        """Test validate_coordinates returns ValidationMetrics."""
        pytest.importorskip("am_qadf_native.correction", reason="Validation C++ bindings required")
        v = CorrectionValidator()
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (2.0, 2.0, 2.0)
        result = v.validate_coordinates(points, bbox_min, bbox_max)
        assert isinstance(result, ValidationMetrics)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.metrics, dict)

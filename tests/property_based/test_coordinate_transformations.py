"""
Property-based tests for coordinate transformations.

Tests mathematical properties like invertibility and roundtrip
using am_qadf.coordinate_systems (CoordinateSystem).
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck

try:
    from am_qadf.coordinate_systems.coordinate_systems import (
        CoordinateSystem,
        CPP_AVAILABLE,
    )
    COORDINATE_TRANSFORM_AVAILABLE = True
except ImportError:
    COORDINATE_TRANSFORM_AVAILABLE = False
    CPP_AVAILABLE = False


# Skip when CoordinateSystem is missing or when C++ bindings (am_qadf_native) are not built
@pytest.mark.skipif(
    not COORDINATE_TRANSFORM_AVAILABLE or not CPP_AVAILABLE,
    reason="CoordinateSystem requires C++ bindings (am_qadf_native); skip when not built",
)
@pytest.mark.property_based
class TestCoordinateTransformationProperties:
    """Property-based tests for coordinate transformations."""

    @given(
        origin=st.tuples(
            st.floats(min_value=-50.0, max_value=50.0),
            st.floats(min_value=-50.0, max_value=50.0),
            st.floats(min_value=-50.0, max_value=50.0),
        ),
        rotation=st.tuples(
            st.floats(min_value=-180.0, max_value=180.0),
            st.floats(min_value=-180.0, max_value=180.0),
            st.floats(min_value=-180.0, max_value=180.0),
        ),
        scale=st.floats(min_value=0.1, max_value=10.0),
        n_points=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=20)
    def test_coordinate_system_roundtrip(self, origin, rotation, scale, n_points):
        """Test that coordinate system transformation is invertible."""
        assume(scale > 0.01)  # Avoid very small scales

        # Create coordinate system (am_qadf.coordinate_systems)
        system = CoordinateSystem(name="test_system", origin=origin, rotation=rotation, scale=scale)

        # Generate random points in local coordinates
        np.random.seed(42)
        local_points = np.random.rand(n_points, 3) * 10.0

        # Transform to global and back
        global_points = np.array([system.transform_point(p) for p in local_points])
        back_local_points = np.array([system.inverse_transform_point(p) for p in global_points])

        # Property: Transform then inverse should give original
        assert np.allclose(local_points, back_local_points, atol=1e-6)

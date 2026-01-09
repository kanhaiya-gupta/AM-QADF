"""
Property-based tests for coordinate transformations.

Tests mathematical properties like invertibility, associativity, and identity.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck

try:
    from am_qadf.synchronization.spatial_transformation import (
        TransformationMatrix,
        TransformationManager,
    )
    from am_qadf.voxelization.coordinate_systems import (
        CoordinateSystem,
        CoordinateSystemRegistry,
    )

    COORDINATE_TRANSFORM_AVAILABLE = True
except ImportError:
    COORDINATE_TRANSFORM_AVAILABLE = False


@pytest.mark.skipif(not COORDINATE_TRANSFORM_AVAILABLE, reason="Coordinate transformation not available")
@pytest.mark.property_based
class TestCoordinateTransformationProperties:
    """Property-based tests for coordinate transformations."""

    @given(
        tx=st.floats(min_value=-100.0, max_value=100.0),
        ty=st.floats(min_value=-100.0, max_value=100.0),
        tz=st.floats(min_value=-100.0, max_value=100.0),
        n_points=st.integers(min_value=1, max_value=20),  # Reduced from 50
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_translation_invertibility(self, tx, ty, tz, n_points):
        """Test that translation transformation is invertible."""
        # Create translation transformation
        trans = TransformationMatrix.translation(tx, ty, tz)
        inv_trans = trans.inverse()

        # Generate random points
        np.random.seed(42)
        points = np.random.rand(n_points, 3) * 100.0

        # Apply transformation and inverse
        transformed = trans.apply(points)
        back_transformed = inv_trans.apply(transformed)

        # Property: Transform then inverse should give original
        assert np.allclose(points, back_transformed, atol=1e-10)

    @given(
        axis=st.sampled_from(["x", "y", "z"]),
        angle=st.floats(min_value=-np.pi, max_value=np.pi),
        n_points=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30)
    def test_rotation_invertibility(self, axis, angle, n_points):
        """Test that rotation transformation is invertible."""
        # Create rotation transformation
        trans = TransformationMatrix.rotation(axis, angle)
        inv_trans = trans.inverse()

        # Generate random points
        np.random.seed(42)
        points = np.random.rand(n_points, 3) * 100.0

        # Apply transformation and inverse
        transformed = trans.apply(points)
        back_transformed = inv_trans.apply(transformed)

        # Property: Transform then inverse should give original
        assert np.allclose(points, back_transformed, atol=1e-6)

    @given(
        tx1=st.floats(min_value=-50.0, max_value=50.0),
        ty1=st.floats(min_value=-50.0, max_value=50.0),
        tz1=st.floats(min_value=-50.0, max_value=50.0),
        tx2=st.floats(min_value=-50.0, max_value=50.0),
        ty2=st.floats(min_value=-50.0, max_value=50.0),
        tz2=st.floats(min_value=-50.0, max_value=50.0),
        n_points=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=20)
    def test_transformation_associativity(self, tx1, ty1, tz1, tx2, ty2, tz2, n_points):
        """Test that transformation composition is associative."""
        # Create two transformations
        trans1 = TransformationMatrix.translation(tx1, ty1, tz1)
        trans2 = TransformationMatrix.translation(tx2, ty2, tz2)

        # Compose: (T1 @ T2) @ points = T1 @ (T2 @ points)
        composed_matrix = trans1.matrix @ trans2.matrix
        composed = TransformationMatrix(matrix=composed_matrix)

        # Generate random points
        np.random.seed(42)
        points = np.random.rand(n_points, 3) * 100.0

        # Apply in different orders
        result1 = composed.apply(points)
        result2 = trans1.apply(trans2.apply(points))

        # Property: Results should be identical (associative)
        assert np.allclose(result1, result2, atol=1e-10)

    @given(n_points=st.integers(min_value=1, max_value=50))
    @settings(max_examples=20)
    def test_identity_transformation(self, n_points):
        """Test that identity transformation leaves points unchanged."""
        # Create identity transformation
        identity = TransformationMatrix.identity()

        # Generate random points
        np.random.seed(42)
        points = np.random.rand(n_points, 3) * 100.0

        # Apply identity
        transformed = identity.apply(points)

        # Property: Identity should not change points
        assert np.allclose(points, transformed, atol=1e-10)

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

        # Create coordinate system
        system = CoordinateSystem(name="test_system", origin=origin, rotation=rotation, scale=scale)

        # Generate random points in local coordinates
        np.random.seed(42)
        local_points = np.random.rand(n_points, 3) * 10.0

        # Transform to global and back
        global_points = np.array([system.transform_point(p) for p in local_points])
        back_local_points = np.array([system.inverse_transform_point(p) for p in global_points])

        # Property: Transform then inverse should give original
        assert np.allclose(local_points, back_local_points, atol=1e-6)

    @given(
        origin1=st.tuples(
            st.floats(min_value=-50.0, max_value=50.0),
            st.floats(min_value=-50.0, max_value=50.0),
            st.floats(min_value=-50.0, max_value=50.0),
        ),
        origin2=st.tuples(
            st.floats(min_value=-50.0, max_value=50.0),
            st.floats(min_value=-50.0, max_value=50.0),
            st.floats(min_value=-50.0, max_value=50.0),
        ),
        n_points=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=20)
    def test_coordinate_system_registry_roundtrip(self, origin1, origin2, n_points):
        """Test that coordinate system registry transformations are invertible."""
        # Create registry
        registry = CoordinateSystemRegistry()

        # Register two coordinate systems (registry.register takes origin, rotation, scale, not a CoordinateSystem object)
        registry.register("system1", origin=origin1, parent=None)
        registry.register("system2", origin=origin2, parent=None)

        # Generate random points in system1
        np.random.seed(42)
        points1 = np.random.rand(n_points, 3) * 10.0

        # Transform to system2 and back
        points2 = np.array([registry.transform(p, "system1", "system2") for p in points1])
        points1_back = np.array([registry.transform(p, "system2", "system1") for p in points2])

        # Property: Transform then inverse should give original
        assert np.allclose(points1, points1_back, atol=1e-6)

    @given(n_points=st.integers(min_value=1, max_value=30))
    @settings(max_examples=10)
    def test_same_system_identity(self, n_points):
        """Test that transforming within the same coordinate system is identity."""
        # Create registry
        registry = CoordinateSystemRegistry()

        # Register a coordinate system (registry.register takes origin, rotation, scale, not a CoordinateSystem object)
        registry.register("system1", origin=(0.0, 0.0, 0.0), parent=None)

        # Generate random points
        np.random.seed(42)
        points = np.random.rand(n_points, 3) * 10.0

        # Transform to same system
        transformed = np.array([registry.transform(p, "system1", "system1") for p in points])

        # Property: Should be identity
        assert np.allclose(points, transformed, atol=1e-10)

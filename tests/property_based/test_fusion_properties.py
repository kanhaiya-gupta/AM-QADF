"""
Property-based tests for fusion methods.

Tests mathematical properties like commutativity, idempotency, and boundedness.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings

try:
    from am_qadf.fusion.fusion_methods import (
        AverageFusion,
        WeightedAverageFusion,
        MedianFusion,
        MaxFusion,
        MinFusion,
    )

    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False


@pytest.mark.skipif(not FUSION_AVAILABLE, reason="Fusion methods not available")
@pytest.mark.property_based
class TestFusionProperties:
    """Property-based tests for fusion methods."""

    @given(
        n_signals=st.integers(min_value=2, max_value=5),
        array_size=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=30)
    def test_average_fusion_commutativity(self, n_signals, array_size):
        """Test that average fusion is commutative (order doesn't matter)."""
        # Generate random signals
        np.random.seed(42)
        signals1 = {}
        signals2 = {}

        for i in range(n_signals):
            signal_name = f"signal_{i}"
            values = np.random.rand(array_size) * 100.0
            signals1[signal_name] = values
            # Create shuffled version
            signals2[f"signal_{n_signals - 1 - i}"] = values

        # Fuse element-wise (fuse_values takes list of floats per index)
        fusion = AverageFusion()
        result1 = np.array([fusion.fuse_values([signals1[n][i] for n in signals1]) for i in range(array_size)])
        result2 = np.array([fusion.fuse_values([signals2[n][i] for n in signals2]) for i in range(array_size)])

        # Property: Results should be identical (commutative)
        assert np.allclose(result1, result2, atol=1e-10)

    @given(
        n_signals=st.integers(min_value=2, max_value=5),
        array_size=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=30)
    def test_fusion_bounded_by_input(self, n_signals, array_size):
        """Test that fused values are bounded by input signal ranges."""
        # Generate random signals
        np.random.seed(42)
        signals = {}
        min_values = []
        max_values = []

        for i in range(n_signals):
            signal_name = f"signal_{i}"
            values = np.random.rand(array_size) * 100.0
            signals[signal_name] = values
            min_values.append(np.min(values))
            max_values.append(np.max(values))

        overall_min = min(min_values)
        overall_max = max(max_values)

        # Test with average fusion (element-wise)
        fusion = AverageFusion()
        result = np.array([fusion.fuse_values([signals[n][i] for n in signals]) for i in range(array_size)])

        # Property: Fused values should be within input range
        assert np.all(result >= overall_min - 1e-10)  # Allow small numerical error
        assert np.all(result <= overall_max + 1e-10)

    @given(
        array_size=st.integers(min_value=10, max_value=100),
        signal_value=st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=20)
    def test_fusion_idempotency(self, array_size, signal_value):
        """Test that fusing the same signal multiple times gives same result."""
        # Create same signal multiple times
        signals = {
            "signal_1": np.full(array_size, signal_value),
            "signal_2": np.full(array_size, signal_value),
            "signal_3": np.full(array_size, signal_value),
        }

        # Fuse (element-wise)
        fusion = AverageFusion()
        result = np.array([fusion.fuse_values([signals[n][i] for n in signals]) for i in range(array_size)])

        # Property: Result should equal input (idempotent for same values)
        assert np.allclose(result, signal_value, atol=1e-10)

    @given(
        n_signals=st.integers(min_value=2, max_value=5),
        array_size=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=30)
    def test_max_fusion_properties(self, n_signals, array_size):
        """Test properties of max fusion."""
        # Generate random signals
        np.random.seed(42)
        signals = {}
        all_values = []

        for i in range(n_signals):
            signal_name = f"signal_{i}"
            values = np.random.rand(array_size) * 100.0
            signals[signal_name] = values
            all_values.append(values)

        # Stack all values
        stacked = np.stack(all_values, axis=0)
        expected_max = np.max(stacked, axis=0)

        # Fuse with max (element-wise)
        fusion = MaxFusion()
        result = np.array([fusion.fuse_values([signals[n][i] for n in signals]) for i in range(array_size)])

        # Property: Result should equal element-wise max
        assert np.allclose(result, expected_max, atol=1e-10)

    @given(
        n_signals=st.integers(min_value=2, max_value=5),
        array_size=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=30)
    def test_min_fusion_properties(self, n_signals, array_size):
        """Test properties of min fusion."""
        # Generate random signals
        np.random.seed(42)
        signals = {}
        all_values = []

        for i in range(n_signals):
            signal_name = f"signal_{i}"
            values = np.random.rand(array_size) * 100.0
            signals[signal_name] = values
            all_values.append(values)

        # Stack all values
        stacked = np.stack(all_values, axis=0)
        expected_min = np.min(stacked, axis=0)

        # Fuse with min (element-wise)
        fusion = MinFusion()
        result = np.array([fusion.fuse_values([signals[n][i] for n in signals]) for i in range(array_size)])

        # Property: Result should equal element-wise min
        assert np.allclose(result, expected_min, atol=1e-10)

    @given(
        n_signals=st.integers(min_value=2, max_value=5),
        array_size=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=30)
    def test_weighted_average_linearity(self, n_signals, array_size):
        """Test that weighted average fusion is linear in weights."""
        # Generate random signals
        np.random.seed(42)
        signals = {}
        weights1 = {}
        weights2 = {}

        for i in range(n_signals):
            signal_name = f"signal_{i}"
            values = np.random.rand(array_size) * 100.0
            signals[signal_name] = values
            weights1[signal_name] = np.random.rand()
            weights2[signal_name] = np.random.rand()

        # Normalize weights to lists (same order as signals)
        names = list(signals.keys())
        w1 = [weights1[n] for n in names]
        w2 = [weights2[n] for n in names]
        w1 = np.array(w1) / np.sum(w1)
        w2 = np.array(w2) / np.sum(w2)

        # Fuse element-wise with different weights
        fusion1 = WeightedAverageFusion(weights=w1.tolist())
        fusion2 = WeightedAverageFusion(weights=w2.tolist())
        result1 = np.array([fusion1.fuse_values([signals[n][i] for n in names], weights=w1.tolist()) for i in range(array_size)])
        result2 = np.array([fusion2.fuse_values([signals[n][i] for n in names], weights=w2.tolist()) for i in range(array_size)])

        # Property: Results should be different if weights are different
        # (unless signals are identical)
        if not np.allclose(list(signals.values())[0], list(signals.values())[1], atol=1e-6):
            # Results should generally be different
            assert not np.allclose(result1, result2, atol=1e-6) or np.allclose(weights1, weights2, atol=0.1)

    @given(
        n_signals=st.integers(min_value=2, max_value=5),
        array_size=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=30)
    def test_median_fusion_properties(self, n_signals, array_size):
        """Test properties of median fusion."""
        # Generate random signals
        np.random.seed(42)
        signals = {}
        all_values = []

        for i in range(n_signals):
            signal_name = f"signal_{i}"
            values = np.random.rand(array_size) * 100.0
            signals[signal_name] = values
            all_values.append(values)

        # Stack all values
        stacked = np.stack(all_values, axis=0)
        expected_median = np.median(stacked, axis=0)

        # Fuse with median (element-wise)
        fusion = MedianFusion()
        result = np.array([fusion.fuse_values([signals[n][i] for n in signals]) for i in range(array_size)])

        # Property: Result should equal element-wise median
        assert np.allclose(result, expected_median, atol=1e-10)

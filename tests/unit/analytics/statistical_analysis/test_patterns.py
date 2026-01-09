"""
Unit tests for pattern recognition.

Tests for PatternResults and PatternAnalyzer.
"""

import pytest
import numpy as np
from am_qadf.analytics.statistical_analysis.patterns import (
    PatternResults,
    PatternAnalyzer,
)


class TestPatternResults:
    """Test suite for PatternResults dataclass."""

    @pytest.mark.unit
    def test_results_creation(self):
        """Test creating PatternResults."""
        spatial_patterns = {"signal1": {"clusters": []}}
        temporal_patterns = {"signal1": {"periodic": False}}

        results = PatternResults(spatial_patterns=spatial_patterns, temporal_patterns=temporal_patterns)

        assert len(results.spatial_patterns) == 1
        assert len(results.temporal_patterns) == 1
        assert results.anomaly_patterns is None
        assert results.process_patterns is None

    @pytest.mark.unit
    def test_results_to_dict(self):
        """Test converting PatternResults to dictionary."""
        spatial_patterns = {"signal1": {"clusters": []}}
        temporal_patterns = {"signal1": {"periodic": False}}

        results = PatternResults(spatial_patterns=spatial_patterns, temporal_patterns=temporal_patterns)

        result_dict = results.to_dict()

        assert isinstance(result_dict, dict)
        assert "spatial_patterns" in result_dict
        assert "temporal_patterns" in result_dict


class TestPatternAnalyzer:
    """Test suite for PatternAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a PatternAnalyzer instance."""
        return PatternAnalyzer()

    @pytest.mark.unit
    def test_analyzer_creation(self, analyzer):
        """Test creating PatternAnalyzer."""
        assert analyzer is not None

    @pytest.mark.unit
    def test_detect_spatial_clusters(self, analyzer):
        """Test detecting spatial clusters."""
        # Create signal array with a cluster
        signal_array = np.zeros((10, 10, 5))
        signal_array[3:7, 3:7, :] = 1.0  # Create a cluster

        clusters = analyzer.detect_spatial_clusters(signal_array, threshold=0.5)

        assert isinstance(clusters, dict)
        assert "num_clusters" in clusters
        assert "clusters" in clusters
        assert clusters["num_clusters"] > 0

    @pytest.mark.unit
    def test_detect_spatial_clusters_no_clusters(self, analyzer):
        """Test detecting clusters when none exist."""
        signal_array = np.zeros((10, 10, 5))

        clusters = analyzer.detect_spatial_clusters(signal_array, threshold=0.5)

        assert clusters["num_clusters"] == 0
        assert len(clusters["clusters"]) == 0

    @pytest.mark.unit
    def test_detect_spatial_clusters_auto_threshold(self, analyzer):
        """Test detecting clusters with automatic threshold."""
        signal_array = np.random.rand(10, 10, 5)
        signal_array[3:7, 3:7, :] = 2.0  # Create a high-value cluster

        clusters = analyzer.detect_spatial_clusters(signal_array, threshold=None)

        assert isinstance(clusters, dict)
        assert "threshold" in clusters

    @pytest.mark.unit
    def test_detect_periodic_patterns(self, analyzer):
        """Test detecting periodic patterns."""
        # Create signal array with periodic pattern along z-axis
        signal_array = np.zeros((5, 5, 20))
        for z in range(20):
            signal_array[:, :, z] = np.sin(z * np.pi / 5)

        patterns = analyzer.detect_periodic_patterns(signal_array, axis=2)

        assert isinstance(patterns, dict)
        assert "has_periodicity" in patterns

    @pytest.mark.unit
    def test_detect_periodic_patterns_no_periodicity(self, analyzer):
        """Test detecting periodic patterns when none exist."""
        signal_array = np.random.rand(5, 5, 20)

        patterns = analyzer.detect_periodic_patterns(signal_array, axis=2)

        assert isinstance(patterns, dict)
        assert "has_periodicity" in patterns

    @pytest.mark.unit
    def test_detect_periodic_patterns_invalid_axis(self, analyzer):
        """Test detecting periodic patterns with invalid axis."""
        signal_array = np.zeros((5, 5, 5))

        patterns = analyzer.detect_periodic_patterns(signal_array, axis=5)

        assert patterns["has_periodicity"] is False

    @pytest.mark.unit
    def test_detect_anomaly_patterns(self, analyzer):
        """Test detecting anomaly patterns."""
        # Create signal array with anomalies
        signal_array = np.ones((10, 10, 5)) * 1.0
        signal_array[5, 5, 2] = 10.0  # Anomaly

        anomalies = analyzer.detect_anomaly_patterns(signal_array, "test_signal")

        assert isinstance(anomalies, dict)
        assert "num_anomalies" in anomalies or "anomalies" in anomalies

    @pytest.mark.unit
    def test_detect_process_patterns(self, analyzer):
        """Test detecting process patterns."""

        class MockVoxelData:
            def get_signal_array(self, signal_name, default=0.0):
                return np.ones((10, 10, 10))

        voxel_data = MockVoxelData()
        patterns = analyzer.detect_process_patterns(voxel_data, ["signal1"])

        assert isinstance(patterns, dict)

    @pytest.mark.unit
    def test_analyze_patterns(self, analyzer):
        """Test comprehensive pattern analysis."""

        class MockVoxelData:
            def __init__(self):
                self.available_signals = ["signal1"]

            def get_signal_array(self, signal_name, default=0.0):
                signal_array = np.zeros((10, 10, 5))
                signal_array[3:7, 3:7, :] = 1.0  # Create a cluster
                return signal_array

        voxel_data = MockVoxelData()
        result = analyzer.analyze_patterns(voxel_data, ["signal1"])

        assert isinstance(result, PatternResults)
        assert "signal1" in result.spatial_patterns
        assert "signal1" in result.temporal_patterns

    @pytest.mark.unit
    def test_analyze_patterns_with_anomalies(self, analyzer):
        """Test pattern analysis with anomaly detection."""

        class MockVoxelData:
            def get_signal_array(self, signal_name, default=0.0):
                signal_array = np.ones((10, 10, 5))
                signal_array[5, 5, 2] = 10.0  # Anomaly
                return signal_array

        voxel_data = MockVoxelData()
        result = analyzer.analyze_patterns(voxel_data, ["signal1"], include_anomalies=True)

        assert isinstance(result, PatternResults)
        assert result.anomaly_patterns is not None

    @pytest.mark.unit
    def test_analyze_patterns_with_process(self, analyzer):
        """Test pattern analysis with process pattern detection."""

        class MockVoxelData:
            def get_signal_array(self, signal_name, default=0.0):
                return np.ones((10, 10, 10))

        voxel_data = MockVoxelData()
        result = analyzer.analyze_patterns(voxel_data, ["signal1"], include_process=True)

        assert isinstance(result, PatternResults)
        assert result.process_patterns is not None

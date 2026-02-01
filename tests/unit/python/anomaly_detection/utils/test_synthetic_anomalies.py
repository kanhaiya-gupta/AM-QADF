"""
Unit tests for synthetic anomaly generation utilities.

Tests for AnomalyInjectionType, AnomalyInjectionConfig, and SyntheticAnomalyGenerator.
"""

import pytest
import numpy as np
from datetime import datetime
from am_qadf.anomaly_detection.utils.synthetic_anomalies import (
    AnomalyInjectionType,
    AnomalyInjectionConfig,
    SyntheticAnomalyGenerator,
)


class TestAnomalyInjectionType:
    """Test suite for AnomalyInjectionType enum."""

    @pytest.mark.unit
    def test_enum_values(self):
        """Test enum values."""
        assert AnomalyInjectionType.POINT_OUTLIER.value == "point_outlier"
        assert AnomalyInjectionType.SPATIAL_CLUSTER.value == "spatial_cluster"
        assert AnomalyInjectionType.TEMPORAL_DRIFT.value == "temporal_drift"
        assert AnomalyInjectionType.SUDDEN_CHANGE.value == "sudden_change"
        assert AnomalyInjectionType.CORRELATION_BREAK.value == "correlation_break"
        assert AnomalyInjectionType.CONTEXTUAL.value == "contextual"

    @pytest.mark.unit
    def test_enum_creation_from_value(self):
        """Test creating enum from value."""
        assert AnomalyInjectionType("point_outlier") == AnomalyInjectionType.POINT_OUTLIER
        assert AnomalyInjectionType("spatial_cluster") == AnomalyInjectionType.SPATIAL_CLUSTER


class TestAnomalyInjectionConfig:
    """Test suite for AnomalyInjectionConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating AnomalyInjectionConfig with default values."""
        config = AnomalyInjectionConfig()

        assert config.anomaly_type == AnomalyInjectionType.POINT_OUTLIER
        assert config.magnitude == 3.0
        assert config.cluster_size == 10
        assert config.cluster_radius == 2.0
        assert config.drift_rate == 0.1
        assert config.change_point is None
        assert config.target_features is None
        assert config.feature_weights is None
        assert config.random_seed is None

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating AnomalyInjectionConfig with custom values."""
        config = AnomalyInjectionConfig(
            anomaly_type=AnomalyInjectionType.SPATIAL_CLUSTER,
            magnitude=5.0,
            cluster_size=20,
            cluster_radius=3.0,
            drift_rate=0.2,
            change_point=50,
            target_features=["laser_power", "density"],
            feature_weights={"laser_power": 0.7, "density": 0.3},
            random_seed=42,
        )

        assert config.anomaly_type == AnomalyInjectionType.SPATIAL_CLUSTER
        assert config.magnitude == 5.0
        assert config.cluster_size == 20
        assert config.cluster_radius == 3.0
        assert config.drift_rate == 0.2
        assert config.change_point == 50
        assert config.target_features == ["laser_power", "density"]
        assert config.feature_weights == {"laser_power": 0.7, "density": 0.3}
        assert config.random_seed == 42


class TestSyntheticAnomalyGenerator:
    """Test suite for SyntheticAnomalyGenerator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample fused voxel data."""

        class MockVoxelData:
            def __init__(self, idx):
                self.laser_power = 100.0 + np.random.randn() * 10
                self.density = 0.9 + np.random.randn() * 0.05
                self.temperature = 200.0 + np.random.randn() * 20
                self.layer_number = idx[2] if len(idx) > 2 else 0
                self.build_time = datetime.now()

        return {(i, j, k): MockVoxelData((i, j, k)) for i in range(10) for j in range(10) for k in range(5)}

    @pytest.mark.unit
    def test_generator_creation_default(self):
        """Test creating SyntheticAnomalyGenerator with default config."""
        generator = SyntheticAnomalyGenerator()

        assert generator.config is not None
        assert generator.config.anomaly_type == AnomalyInjectionType.POINT_OUTLIER

    @pytest.mark.unit
    def test_generator_creation_custom_config(self):
        """Test creating SyntheticAnomalyGenerator with custom config."""
        config = AnomalyInjectionConfig(anomaly_type=AnomalyInjectionType.SPATIAL_CLUSTER, magnitude=5.0)
        generator = SyntheticAnomalyGenerator(config=config)

        assert generator.config.anomaly_type == AnomalyInjectionType.SPATIAL_CLUSTER
        assert generator.config.magnitude == 5.0

    @pytest.mark.unit
    def test_generator_with_random_seed(self):
        """Test generator with random seed."""
        config = AnomalyInjectionConfig(random_seed=42)
        generator = SyntheticAnomalyGenerator(config=config)

        assert generator.config.random_seed == 42

    @pytest.mark.unit
    def test_inject_point_outliers(self, sample_data):
        """Test injecting point outlier anomalies."""
        config = AnomalyInjectionConfig(anomaly_type=AnomalyInjectionType.POINT_OUTLIER, magnitude=3.0)
        generator = SyntheticAnomalyGenerator(config=config)

        modified_data, labels = generator.inject_anomalies(sample_data, n_anomalies=10)

        assert len(modified_data) == len(sample_data)
        assert len(labels) == len(sample_data)
        num_anomalies = sum(labels.values())
        assert num_anomalies == 10

    @pytest.mark.unit
    def test_inject_spatial_clusters(self, sample_data):
        """Test injecting spatial cluster anomalies."""
        config = AnomalyInjectionConfig(
            anomaly_type=AnomalyInjectionType.SPATIAL_CLUSTER,
            cluster_size=5,
            cluster_radius=2.0,
        )
        generator = SyntheticAnomalyGenerator(config=config)

        modified_data, labels = generator.inject_anomalies(sample_data, n_anomalies=3)

        assert len(modified_data) == len(sample_data)
        num_anomalies = sum(labels.values())
        assert num_anomalies > 0

    @pytest.mark.unit
    def test_inject_temporal_drift(self, sample_data):
        """Test injecting temporal drift anomalies."""
        config = AnomalyInjectionConfig(anomaly_type=AnomalyInjectionType.TEMPORAL_DRIFT, drift_rate=0.1)
        generator = SyntheticAnomalyGenerator(config=config)

        modified_data, labels = generator.inject_anomalies(sample_data)

        assert len(modified_data) == len(sample_data)
        num_anomalies = sum(labels.values())
        assert num_anomalies > 0

    @pytest.mark.unit
    def test_inject_sudden_change(self, sample_data):
        """Test injecting sudden change anomalies."""
        config = AnomalyInjectionConfig(
            anomaly_type=AnomalyInjectionType.SUDDEN_CHANGE,
            change_point=100,
            magnitude=3.0,
        )
        generator = SyntheticAnomalyGenerator(config=config)

        modified_data, labels = generator.inject_anomalies(sample_data)

        assert len(modified_data) == len(sample_data)
        num_anomalies = sum(labels.values())
        assert num_anomalies > 0

    @pytest.mark.unit
    def test_inject_correlation_break(self, sample_data):
        """Test injecting correlation break anomalies."""
        config = AnomalyInjectionConfig(anomaly_type=AnomalyInjectionType.CORRELATION_BREAK, magnitude=2.0)
        generator = SyntheticAnomalyGenerator(config=config)

        modified_data, labels = generator.inject_anomalies(sample_data, n_anomalies=10)

        assert len(modified_data) == len(sample_data)
        num_anomalies = sum(labels.values())
        assert num_anomalies == 10

    @pytest.mark.unit
    def test_inject_with_target_features(self, sample_data):
        """Test injecting anomalies with target features."""
        config = AnomalyInjectionConfig(
            anomaly_type=AnomalyInjectionType.POINT_OUTLIER,
            target_features=["laser_power", "density"],
        )
        generator = SyntheticAnomalyGenerator(config=config)

        modified_data, labels = generator.inject_anomalies(sample_data, n_anomalies=5)

        assert len(modified_data) == len(sample_data)
        num_anomalies = sum(labels.values())
        assert num_anomalies == 5

    @pytest.mark.unit
    def test_inject_with_existing_labels(self, sample_data):
        """Test injecting anomalies with existing labels."""
        existing_labels = {(i, j, k): False for i, j, k in sample_data.keys()}
        existing_labels[(0, 0, 0)] = True  # Mark one as already anomalous

        config = AnomalyInjectionConfig(anomaly_type=AnomalyInjectionType.POINT_OUTLIER)
        generator = SyntheticAnomalyGenerator(config=config)

        modified_data, labels = generator.inject_anomalies(sample_data, n_anomalies=5, anomaly_labels=existing_labels)

        assert len(labels) == len(sample_data)
        # Should have at least 5 new anomalies (plus the existing one)
        num_anomalies = sum(labels.values())
        assert num_anomalies >= 5

    @pytest.mark.unit
    def test_calculate_feature_statistics(self, sample_data):
        """Test calculating feature statistics."""
        generator = SyntheticAnomalyGenerator()
        stats = generator._calculate_feature_statistics(sample_data)

        assert isinstance(stats, dict)
        assert len(stats) > 0
        for feature_name, (mean, std) in stats.items():
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert std >= 0

    @pytest.mark.unit
    def test_calculate_feature_correlations(self, sample_data):
        """Test calculating feature correlations."""
        generator = SyntheticAnomalyGenerator()
        correlations = generator._calculate_feature_correlations(sample_data)

        assert isinstance(correlations, dict)
        assert len(correlations) > 0

    @pytest.mark.unit
    def test_get_temporal_key(self, sample_data):
        """Test getting temporal key for sorting."""
        generator = SyntheticAnomalyGenerator()

        # Test with layer_number
        first_item = next(iter(sample_data.values()))
        key = generator._get_temporal_key(first_item)
        assert isinstance(key, float)

        # Test with build_time
        key2 = generator._get_temporal_key(first_item)
        assert isinstance(key2, float)

    @pytest.mark.unit
    def test_inject_more_anomalies_than_data(self, sample_data):
        """Test injecting more anomalies than available data points."""
        config = AnomalyInjectionConfig(anomaly_type=AnomalyInjectionType.POINT_OUTLIER)
        generator = SyntheticAnomalyGenerator(config=config)

        # Try to inject more anomalies than data points
        modified_data, labels = generator.inject_anomalies(sample_data, n_anomalies=10000)

        # Should not fail, but limit to available data
        assert len(modified_data) == len(sample_data)
        num_anomalies = sum(labels.values())
        assert num_anomalies <= len(sample_data)

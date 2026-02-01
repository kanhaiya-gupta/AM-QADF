"""
Unit tests for base anomaly detector.

Tests for BaseAnomalyDetector, AnomalyDetectionConfig, and AnomalyDetectionResult.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
    BaseAnomalyDetector,
)
from am_qadf.anomaly_detection.core.types import AnomalyType


class TestAnomalyDetectionConfig:
    """Test suite for AnomalyDetectionConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation_default(self):
        """Test creating AnomalyDetectionConfig with default values."""
        config = AnomalyDetectionConfig()

        assert config.threshold == 0.5
        assert config.confidence_threshold == 0.7
        assert config.normalize_features is True
        assert config.handle_missing == "mean"
        assert config.remove_outliers is False
        assert config.parallel_processing is False
        assert config.max_workers == 4
        assert config.batch_size == 1000
        assert isinstance(config.method_params, dict)
        assert len(config.method_params) == 0

    @pytest.mark.unit
    def test_config_creation_custom(self):
        """Test creating AnomalyDetectionConfig with custom values."""
        method_params = {"n_neighbors": 5, "contamination": 0.1}
        config = AnomalyDetectionConfig(
            threshold=0.7,
            confidence_threshold=0.8,
            normalize_features=False,
            handle_missing="median",
            remove_outliers=True,
            parallel_processing=True,
            max_workers=8,
            batch_size=2000,
            method_params=method_params,
        )

        assert config.threshold == 0.7
        assert config.confidence_threshold == 0.8
        assert config.normalize_features is False
        assert config.handle_missing == "median"
        assert config.remove_outliers is True
        assert config.parallel_processing is True
        assert config.max_workers == 8
        assert config.batch_size == 2000
        assert config.method_params == method_params

    @pytest.mark.unit
    def test_config_handle_missing_options(self):
        """Test different handle_missing options."""
        for option in ["mean", "median", "drop", "zero"]:
            config = AnomalyDetectionConfig(handle_missing=option)
            assert config.handle_missing == option

    @pytest.mark.unit
    def test_config_method_params(self):
        """Test method_params dictionary."""
        params = {"param1": 1.0, "param2": "value", "param3": [1, 2, 3]}
        config = AnomalyDetectionConfig(method_params=params)

        assert config.method_params == params
        assert config.method_params["param1"] == 1.0
        assert config.method_params["param2"] == "value"


class TestAnomalyDetectionResult:
    """Test suite for AnomalyDetectionResult dataclass."""

    @pytest.mark.unit
    def test_result_creation_minimal(self):
        """Test creating AnomalyDetectionResult with minimal required fields."""
        result = AnomalyDetectionResult(
            voxel_index=(0, 0, 0),
            voxel_coordinates=(0.0, 0.0, 0.0),
            is_anomaly=False,
            anomaly_score=0.3,
            confidence=0.8,
            detector_name="TestDetector",
        )

        assert result.voxel_index == (0, 0, 0)
        assert result.voxel_coordinates == (0.0, 0.0, 0.0)
        assert result.is_anomaly is False
        assert result.anomaly_score == 0.3
        assert result.confidence == 0.8
        assert result.detector_name == "TestDetector"
        assert result.anomaly_type is None
        assert isinstance(result.detection_timestamp, datetime)
        assert isinstance(result.features, dict)
        assert isinstance(result.metadata, dict)

    @pytest.mark.unit
    def test_result_creation_full(self):
        """Test creating AnomalyDetectionResult with all fields."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        features = {"feature1": 1.0, "feature2": 2.0}
        metadata = {"key": "value"}

        result = AnomalyDetectionResult(
            voxel_index=(5, 10, 15),
            voxel_coordinates=(1.5, 2.0, 2.5),
            is_anomaly=True,
            anomaly_score=0.9,
            confidence=0.95,
            detector_name="TestDetector",
            anomaly_type=AnomalyType.SPATIAL,
            detection_timestamp=timestamp,
            features=features,
            metadata=metadata,
        )

        assert result.voxel_index == (5, 10, 15)
        assert result.voxel_coordinates == (1.5, 2.0, 2.5)
        assert result.is_anomaly is True
        assert result.anomaly_score == 0.9
        assert result.confidence == 0.95
        assert result.detector_name == "TestDetector"
        assert result.anomaly_type == AnomalyType.SPATIAL
        assert result.detection_timestamp == timestamp
        assert result.features == features
        assert result.metadata == metadata

    @pytest.mark.unit
    def test_result_anomaly_types(self):
        """Test AnomalyDetectionResult with different anomaly types."""
        for anomaly_type in AnomalyType:
            result = AnomalyDetectionResult(
                voxel_index=(0, 0, 0),
                voxel_coordinates=(0.0, 0.0, 0.0),
                is_anomaly=True,
                anomaly_score=0.8,
                confidence=0.9,
                detector_name="TestDetector",
                anomaly_type=anomaly_type,
            )
            assert result.anomaly_type == anomaly_type

    @pytest.mark.unit
    def test_result_default_timestamp(self):
        """Test that default timestamp is created automatically."""
        result1 = AnomalyDetectionResult(
            voxel_index=(0, 0, 0),
            voxel_coordinates=(0.0, 0.0, 0.0),
            is_anomaly=False,
            anomaly_score=0.3,
            confidence=0.8,
            detector_name="TestDetector",
        )

        # Small delay
        import time

        time.sleep(0.01)

        result2 = AnomalyDetectionResult(
            voxel_index=(0, 0, 0),
            voxel_coordinates=(0.0, 0.0, 0.0),
            is_anomaly=False,
            anomaly_score=0.3,
            confidence=0.8,
            detector_name="TestDetector",
        )

        assert result1.detection_timestamp <= result2.detection_timestamp


class TestBaseAnomalyDetector:
    """Test suite for BaseAnomalyDetector abstract class."""

    @pytest.mark.unit
    def test_cannot_instantiate_base_class(self):
        """Test that BaseAnomalyDetector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAnomalyDetector()

    @pytest.mark.unit
    def test_concrete_detector_initialization(self):
        """Test that a concrete detector can be initialized."""

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                self.is_fitted = True
                return self

            def predict(self, data):
                return []

        detector = ConcreteDetector()

        assert detector.config is not None
        assert isinstance(detector.config, AnomalyDetectionConfig)
        assert detector.name == "ConcreteDetector"
        assert detector.is_fitted is False
        assert isinstance(detector.feature_names, list)

    @pytest.mark.unit
    def test_detector_with_custom_config(self):
        """Test detector initialization with custom config."""

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                return self

            def predict(self, data):
                return []

        custom_config = AnomalyDetectionConfig(threshold=0.8)
        detector = ConcreteDetector(config=custom_config)

        assert detector.config.threshold == 0.8
        assert detector.config is custom_config

    @pytest.mark.unit
    def test_preprocess_data_numpy_array(self):
        """Test preprocessing numpy array data."""

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                return self

            def predict(self, data):
                return []

        detector = ConcreteDetector()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        processed = detector._preprocess_data(data)

        assert isinstance(processed, np.ndarray)
        assert processed.shape == data.shape

    @pytest.mark.unit
    def test_preprocess_data_dataframe(self):
        """Test preprocessing pandas DataFrame."""

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                return self

            def predict(self, data):
                return []

        detector = ConcreteDetector()
        data = pd.DataFrame({"col1": [1.0, 2.0], "col2": [3.0, 4.0]})

        processed = detector._preprocess_data(data)

        assert isinstance(processed, np.ndarray)
        assert processed.shape == (2, 2)

    @pytest.mark.unit
    def test_preprocess_data_with_missing_mean(self):
        """Test preprocessing with missing values using mean."""

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                return self

            def predict(self, data):
                return []

        config = AnomalyDetectionConfig(handle_missing="mean")
        detector = ConcreteDetector(config=config)
        data = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])

        processed = detector._preprocess_data(data)

        assert not np.isnan(processed).any()
        assert processed.shape == data.shape

    @pytest.mark.unit
    def test_preprocess_data_with_missing_median(self):
        """Test preprocessing with missing values using median."""

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                return self

            def predict(self, data):
                return []

        config = AnomalyDetectionConfig(handle_missing="median")
        detector = ConcreteDetector(config=config)
        data = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])

        processed = detector._preprocess_data(data)

        assert not np.isnan(processed).any()

    @pytest.mark.unit
    def test_preprocess_data_with_missing_zero(self):
        """Test preprocessing with missing values using zero."""

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                return self

            def predict(self, data):
                return []

        # Disable normalization to test zero-filling in isolation
        config = AnomalyDetectionConfig(handle_missing="zero", normalize_features=False)
        detector = ConcreteDetector(config=config)
        data = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])

        processed = detector._preprocess_data(data)

        assert not np.isnan(processed).any()
        assert processed[1, 0] == 0.0

    @pytest.mark.unit
    def test_preprocess_data_normalization(self):
        """Test feature normalization."""

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                self.is_fitted = True
                return self

            def predict(self, data):
                return []

        config = AnomalyDetectionConfig(normalize_features=True)
        detector = ConcreteDetector(config=config)
        data = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

        # Fit first
        detector.fit(data)

        # Then predict (normalization should be applied)
        processed = detector._preprocess_data(data)

        # Check that data is normalized (mean ~0, std ~1)
        assert processed.shape == data.shape
        # Mean should be close to 0 after normalization
        assert np.abs(np.mean(processed, axis=0)).max() < 1e-10

    @pytest.mark.unit
    def test_calculate_confidence(self):
        """Test confidence calculation."""

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                return self

            def predict(self, data):
                return []

        detector = ConcreteDetector()
        scores = np.array([0.3, 0.5, 0.7, 0.9])
        threshold = 0.5

        confidence = detector._calculate_confidence(scores, threshold)

        assert len(confidence) == len(scores)
        assert np.all(confidence >= 0.0)
        assert np.all(confidence <= 1.0)
        # Score at threshold should have lower confidence
        assert confidence[1] < confidence[3]  # 0.5 < 0.9

    @pytest.mark.unit
    def test_create_results(self):
        """Test creating AnomalyDetectionResult objects."""

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                return self

            def predict(self, data):
                return []

        config = AnomalyDetectionConfig(threshold=0.5)
        detector = ConcreteDetector(config=config)

        scores = np.array([0.3, 0.6, 0.8])
        indices = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        coordinates = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]

        results = detector._create_results(scores, indices, coordinates)

        assert len(results) == 3
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)
        assert results[0].is_anomaly is False  # 0.3 < 0.5
        assert results[1].is_anomaly is True  # 0.6 >= 0.5
        assert results[2].is_anomaly is True  # 0.8 >= 0.5
        assert results[0].voxel_index == (0, 0, 0)
        assert results[1].voxel_index == (1, 1, 1)
        assert results[2].voxel_index == (2, 2, 2)

    @pytest.mark.unit
    def test_predict_scores(self):
        """Test predict_scores method."""

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                return self

            def predict(self, data):
                # Return mock results
                return [
                    AnomalyDetectionResult(
                        voxel_index=(0, 0, 0),
                        voxel_coordinates=(0.0, 0.0, 0.0),
                        is_anomaly=False,
                        anomaly_score=0.3,
                        confidence=0.8,
                        detector_name="Test",
                    ),
                    AnomalyDetectionResult(
                        voxel_index=(1, 1, 1),
                        voxel_coordinates=(1.0, 1.0, 1.0),
                        is_anomaly=True,
                        anomaly_score=0.8,
                        confidence=0.9,
                        detector_name="Test",
                    ),
                ]

        detector = ConcreteDetector()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        scores = detector.predict_scores(data)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2
        assert scores[0] == 0.3
        assert scores[1] == 0.8

    @pytest.mark.unit
    def test_dict_to_array_empty(self):
        """Test converting empty dictionary to array."""

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                return self

            def predict(self, data):
                return []

        detector = ConcreteDetector()
        data = {}

        array = detector._dict_to_array(data)

        assert isinstance(array, np.ndarray)
        assert array.size == 0

    @pytest.mark.unit
    def test_dict_to_array_with_features(self):
        """Test converting dictionary with feature objects to array."""

        class MockFeature:
            def __init__(self, val1, val2):
                self.feature1 = val1
                self.feature2 = val2

        class ConcreteDetector(BaseAnomalyDetector):
            def fit(self, data, labels=None):
                return self

            def predict(self, data):
                return []

        detector = ConcreteDetector()
        data = {(0, 0, 0): MockFeature(1.0, 2.0), (1, 1, 1): MockFeature(3.0, 4.0)}

        array = detector._dict_to_array(data)

        assert isinstance(array, np.ndarray)
        assert array.shape == (2, 2)
        assert array[0, 0] == 1.0
        assert array[0, 1] == 2.0
        assert array[1, 0] == 3.0
        assert array[1, 1] == 4.0

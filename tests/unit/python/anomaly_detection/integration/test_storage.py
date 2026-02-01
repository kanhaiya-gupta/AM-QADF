"""
Unit tests for anomaly storage.

Tests for AnomalyResult and AnomalyStorage classes.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from datetime import datetime
from am_qadf.anomaly_detection.integration.storage import (
    AnomalyResult,
    AnomalyStorage,
)


class TestAnomalyResult:
    """Test suite for AnomalyResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating AnomalyResult."""
        result = AnomalyResult(
            model_id="model1",
            detection_id="det1",
            methods=["isolation_forest", "dbscan"],
            anomalies={
                "isolation_forest": [{"is_anomaly": True, "anomaly_score": 0.9}],
                "dbscan": [{"is_anomaly": False, "anomaly_score": 0.2}],
            },
        )

        assert result.model_id == "model1"
        assert result.detection_id == "det1"
        assert len(result.methods) == 2
        assert len(result.anomalies) == 2
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)
        assert result.anomaly_patterns is None
        assert result.detection_metrics is None
        assert result.root_cause_analysis is None

    @pytest.mark.unit
    def test_result_creation_with_timestamp(self):
        """Test creating AnomalyResult with explicit timestamp."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        result = AnomalyResult(
            model_id="model1",
            detection_id="det1",
            methods=["isolation_forest"],
            anomalies={"isolation_forest": []},
            timestamp=timestamp,
        )

        assert result.timestamp == timestamp

    @pytest.mark.unit
    def test_result_creation_with_all_fields(self):
        """Test creating AnomalyResult with all optional fields."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        patterns = {"spatial_clusters": [{"cluster_id": 1}]}
        metrics = {"precision": 0.9, "recall": 0.8}
        rca = {"root_cause": "temperature_spike"}

        result = AnomalyResult(
            model_id="model1",
            detection_id="det1",
            methods=["isolation_forest"],
            anomalies={"isolation_forest": []},
            timestamp=timestamp,
            anomaly_patterns=patterns,
            detection_metrics=metrics,
            root_cause_analysis=rca,
        )

        assert result.anomaly_patterns == patterns
        assert result.detection_metrics == metrics
        assert result.root_cause_analysis == rca

    @pytest.mark.unit
    def test_result_to_dict(self):
        """Test converting AnomalyResult to dictionary."""
        result = AnomalyResult(
            model_id="model1",
            detection_id="det1",
            methods=["isolation_forest"],
            anomalies={"isolation_forest": []},
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["model_id"] == "model1"
        assert result_dict["detection_id"] == "det1"
        assert isinstance(result_dict["timestamp"], str)  # ISO format string

    @pytest.mark.unit
    def test_result_to_dict_with_numpy_arrays(self):
        """Test converting AnomalyResult with numpy arrays to dictionary."""
        result = AnomalyResult(
            model_id="model1",
            detection_id="det1",
            methods=["isolation_forest"],
            anomalies={"isolation_forest": []},
            detection_metrics={"scores": np.array([0.1, 0.2, 0.3])},
        )

        result_dict = result.to_dict()

        # Numpy arrays should be converted to lists
        assert isinstance(result_dict["detection_metrics"]["scores"], list)


class TestAnomalyStorage:
    """Test suite for AnomalyStorage class."""

    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mock MongoDB client."""
        mock_client = Mock()
        mock_client.connected = True

        # Mock collection
        mock_collection = Mock()
        mock_collection.replace_one = Mock(return_value=Mock(upserted_id="det1"))
        mock_client.get_collection = Mock(return_value=mock_collection)
        return mock_client

    @pytest.fixture
    def storage(self, mock_mongo_client):
        """Create an AnomalyStorage instance."""
        return AnomalyStorage(mock_mongo_client)

    @pytest.fixture
    def sample_anomaly_result(self):
        """Create a sample AnomalyResult."""
        return AnomalyResult(
            model_id="model1",
            detection_id="det1",
            methods=["isolation_forest"],
            anomalies={"isolation_forest": []},
        )

    @pytest.mark.unit
    def test_storage_creation(self, mock_mongo_client):
        """Test creating AnomalyStorage."""
        storage = AnomalyStorage(mock_mongo_client)

        assert storage.mongo_client is not None
        assert storage.collection_name == "anomaly_detections"

    @pytest.mark.unit
    def test_store_anomaly_result(self, storage, sample_anomaly_result):
        """Test storing anomaly result."""
        doc_id = storage.store_anomaly_result(sample_anomaly_result)

        assert doc_id == "det1"
        storage.mongo_client.get_collection.assert_called()
        collection = storage.mongo_client.get_collection.return_value
        collection.replace_one.assert_called_once()

    @pytest.mark.unit
    def test_store_anomaly_result_not_connected(self):
        """Test storing when MongoDB is not connected."""
        mock_client = Mock()
        mock_client.connected = False
        storage = AnomalyStorage(mock_client)

        result = AnomalyResult(
            model_id="model1",
            detection_id="det1",
            methods=["isolation_forest"],
            anomalies={"isolation_forest": []},
        )

        with pytest.raises(ConnectionError):
            storage.store_anomaly_result(result)

    @pytest.mark.unit
    def test_store_anomaly_patterns(self, storage):
        """Test storing anomaly patterns."""
        patterns = {
            "spatial_clusters": [{"cluster_id": 1, "voxels": [1, 2, 3]}],
            "temporal_patterns": [{"pattern_type": "spike", "layers": [10, 11, 12]}],
        }

        doc_id = storage.store_anomaly_patterns(model_id="model1", pattern_id="pattern1", patterns=patterns)

        assert doc_id == "pattern1"
        storage.mongo_client.get_collection.assert_called()
        # Should use "anomaly_patterns" collection
        assert "anomaly_patterns" in str(storage.mongo_client.get_collection.call_args)

    @pytest.mark.unit
    def test_store_anomaly_patterns_not_connected(self):
        """Test storing patterns when MongoDB is not connected."""
        mock_client = Mock()
        mock_client.connected = False
        storage = AnomalyStorage(mock_client)

        with pytest.raises(ConnectionError):
            storage.store_anomaly_patterns(model_id="model1", pattern_id="pattern1", patterns={})

    @pytest.mark.unit
    def test_store_detection_metrics(self, storage):
        """Test storing detection metrics."""
        metrics = {
            "precision": 0.85,
            "recall": 0.90,
            "f1_score": 0.87,
            "accuracy": 0.92,
        }

        doc_id = storage.store_detection_metrics(detection_id="det1", metrics=metrics)

        assert doc_id == "det1"
        storage.mongo_client.get_collection.assert_called()
        # Should use "detection_metrics" collection
        assert "detection_metrics" in str(storage.mongo_client.get_collection.call_args)

    @pytest.mark.unit
    def test_store_detection_metrics_not_connected(self):
        """Test storing metrics when MongoDB is not connected."""
        mock_client = Mock()
        mock_client.connected = False
        storage = AnomalyStorage(mock_client)

        with pytest.raises(ConnectionError):
            storage.store_detection_metrics(detection_id="det1", metrics={"precision": 0.9})

    @pytest.mark.unit
    def test_store_anomaly_result_upsert(self, storage, sample_anomaly_result):
        """Test that storing uses upsert (insert or update)."""
        # Store first time
        doc_id1 = storage.store_anomaly_result(sample_anomaly_result)

        # Store again (should update)
        doc_id2 = storage.store_anomaly_result(sample_anomaly_result)

        assert doc_id1 == doc_id2
        # Should be called twice
        collection = storage.mongo_client.get_collection.return_value
        assert collection.replace_one.call_count == 2

    @pytest.mark.unit
    def test_store_multiple_results(self, storage):
        """Test storing multiple anomaly results."""
        results = [
            AnomalyResult(
                model_id="model1",
                detection_id=f"det{i}",
                methods=["isolation_forest"],
                anomalies={"isolation_forest": []},
            )
            for i in range(3)
        ]

        doc_ids = [storage.store_anomaly_result(r) for r in results]

        assert len(doc_ids) == 3
        assert doc_ids == ["det0", "det1", "det2"]

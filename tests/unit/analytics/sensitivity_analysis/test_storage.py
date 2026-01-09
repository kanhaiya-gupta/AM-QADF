"""
Unit tests for sensitivity analysis storage.

Tests for SensitivityResult and SensitivityStorage classes.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from am_qadf.analytics.sensitivity_analysis.storage import (
    SensitivityResult,
    SensitivityStorage,
)


class TestSensitivityResult:
    """Test suite for SensitivityResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating SensitivityResult."""
        result = SensitivityResult(
            model_id="model1",
            analysis_id="analysis1",
            method="sobol",
            parameter_names=["param1", "param2"],
            sensitivity_indices={"S1_param1": 0.5, "S1_param2": 0.3},
            confidence_intervals={"S1_param1": (0.4, 0.6)},
            parameter_bounds={"param1": (0.0, 10.0)},
            sample_size=1000,
        )

        assert result.model_id == "model1"
        assert result.analysis_id == "analysis1"
        assert result.method == "sobol"
        assert len(result.parameter_names) == 2
        assert len(result.sensitivity_indices) == 2
        assert result.sample_size == 1000
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    @pytest.mark.unit
    def test_result_creation_with_timestamp(self):
        """Test creating SensitivityResult with custom timestamp."""
        custom_timestamp = datetime(2023, 1, 1, 12, 0, 0)
        result = SensitivityResult(
            model_id="model1",
            analysis_id="analysis1",
            method="sobol",
            parameter_names=["param1"],
            sensitivity_indices={"S1_param1": 0.5},
            timestamp=custom_timestamp,
        )

        assert result.timestamp == custom_timestamp

    @pytest.mark.unit
    def test_result_to_dict(self):
        """Test converting SensitivityResult to dictionary."""
        result = SensitivityResult(
            model_id="model1",
            analysis_id="analysis1",
            method="sobol",
            parameter_names=["param1"],
            sensitivity_indices={"S1_param1": 0.5},
            confidence_intervals={"S1_param1": (0.4, 0.6)},
            parameter_bounds={"param1": (0.0, 10.0)},
            sample_size=1000,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["model_id"] == "model1"
        assert result_dict["analysis_id"] == "analysis1"
        assert result_dict["method"] == "sobol"
        assert isinstance(result_dict["timestamp"], str)  # Should be ISO string
        assert "_id" not in result_dict  # to_dict doesn't add _id


class TestSensitivityStorage:
    """Test suite for SensitivityStorage class."""

    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mock MongoDB client."""
        client = Mock()
        client.connected = True
        client.get_collection = Mock(return_value=Mock())
        return client

    @pytest.fixture
    def storage(self, mock_mongo_client):
        """Create a SensitivityStorage instance."""
        return SensitivityStorage(mock_mongo_client)

    @pytest.fixture
    def sensitivity_result(self):
        """Create a SensitivityResult for testing."""
        return SensitivityResult(
            model_id="model1",
            analysis_id="analysis1",
            method="sobol",
            parameter_names=["param1", "param2"],
            sensitivity_indices={"S1_param1": 0.5, "S1_param2": 0.3},
            confidence_intervals={"S1_param1": (0.4, 0.6)},
            parameter_bounds={"param1": (0.0, 10.0), "param2": (0.0, 10.0)},
            sample_size=1000,
        )

    @pytest.mark.unit
    def test_storage_creation(self, mock_mongo_client):
        """Test creating SensitivityStorage."""
        storage = SensitivityStorage(mock_mongo_client)

        assert storage.mongo_client == mock_mongo_client
        assert storage.collection_name == "sensitivity_results"

    @pytest.mark.unit
    def test_store_sensitivity_result(self, storage, mock_mongo_client, sensitivity_result):
        """Test storing sensitivity analysis result."""
        mock_collection = Mock()
        mock_collection.replace_one = Mock()
        mock_mongo_client.get_collection.return_value = mock_collection

        result_id = storage.store_sensitivity_result(sensitivity_result)

        assert result_id == "analysis1"
        mock_collection.replace_one.assert_called_once()
        call_args = mock_collection.replace_one.call_args
        assert call_args[0][0] == {"_id": "analysis1"}
        assert call_args[1]["upsert"] is True

    @pytest.mark.unit
    def test_store_sensitivity_result_not_connected(self, mock_mongo_client, sensitivity_result):
        """Test storing when MongoDB client is not connected."""
        mock_mongo_client.connected = False
        storage = SensitivityStorage(mock_mongo_client)

        with pytest.raises(ConnectionError, match="MongoDB client not connected"):
            storage.store_sensitivity_result(sensitivity_result)

    @pytest.mark.unit
    def test_store_doe_design(self, storage, mock_mongo_client):
        """Test storing Design of Experiments design."""
        mock_collection = Mock()
        mock_collection.replace_one = Mock()
        mock_mongo_client.get_collection.return_value = mock_collection

        design_data = {
            "design_type": "factorial",
            "design_matrix": [[0, 0], [1, 0], [0, 1], [1, 1]],
            "parameter_names": ["param1", "param2"],
        }

        design_id = storage.store_doe_design("model1", "design1", design_data)

        assert design_id == "design1"
        mock_collection.replace_one.assert_called_once()
        call_args = mock_collection.replace_one.call_args
        assert call_args[0][0] == {"_id": "design1"}
        assert call_args[1]["upsert"] is True

    @pytest.mark.unit
    def test_store_doe_design_not_connected(self, mock_mongo_client):
        """Test storing DoE design when MongoDB client is not connected."""
        mock_mongo_client.connected = False
        storage = SensitivityStorage(mock_mongo_client)

        with pytest.raises(ConnectionError, match="MongoDB client not connected"):
            storage.store_doe_design("model1", "design1", {})

    @pytest.mark.unit
    def test_store_influence_rankings(self, storage, mock_mongo_client):
        """Test storing process variable influence rankings."""
        mock_collection = Mock()
        mock_collection.replace_one = Mock()
        mock_mongo_client.get_collection.return_value = mock_collection

        rankings = {"param1": 0.8, "param2": 0.5, "param3": 0.3}

        ranking_id = storage.store_influence_rankings("model1", "ranking1", rankings)

        assert ranking_id == "ranking1"
        mock_collection.replace_one.assert_called_once()
        call_args = mock_collection.replace_one.call_args
        assert call_args[0][0] == {"_id": "ranking1"}
        assert call_args[1]["upsert"] is True
        stored_doc = call_args[0][1]
        assert stored_doc["rankings"] == rankings
        assert stored_doc["model_id"] == "model1"

    @pytest.mark.unit
    def test_store_influence_rankings_not_connected(self, mock_mongo_client):
        """Test storing influence rankings when MongoDB client is not connected."""
        mock_mongo_client.connected = False
        storage = SensitivityStorage(mock_mongo_client)

        with pytest.raises(ConnectionError, match="MongoDB client not connected"):
            storage.store_influence_rankings("model1", "ranking1", {})

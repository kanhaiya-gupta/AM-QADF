"""
Unit tests for experiment storage.

Tests for ExperimentResult and ExperimentStorage classes.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime
from am_qadf.analytics.virtual_experiments.storage import (
    ExperimentResult,
    ExperimentStorage,
)


class TestExperimentResult:
    """Test suite for ExperimentResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating ExperimentResult."""
        result = ExperimentResult(
            experiment_id="exp1",
            model_id="model1",
            design_data={"design_type": "factorial", "num_samples": 100},
            results={"quality": 0.8, "density": 0.95},
        )

        assert result.experiment_id == "exp1"
        assert result.model_id == "model1"
        assert result.design_data["design_type"] == "factorial"
        assert result.results["quality"] == 0.8
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    @pytest.mark.unit
    def test_result_creation_with_timestamp(self):
        """Test creating ExperimentResult with explicit timestamp."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        result = ExperimentResult(
            experiment_id="exp1",
            model_id="model1",
            design_data={},
            results={},
            timestamp=timestamp,
        )

        assert result.timestamp == timestamp

    @pytest.mark.unit
    def test_result_to_dict(self):
        """Test converting ExperimentResult to dictionary."""
        result = ExperimentResult(
            experiment_id="exp1",
            model_id="model1",
            design_data={"design_type": "factorial"},
            results={"quality": 0.8},
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["experiment_id"] == "exp1"
        assert result_dict["model_id"] == "model1"
        assert isinstance(result_dict["timestamp"], str)  # ISO format string


class TestExperimentStorage:
    """Test suite for ExperimentStorage class."""

    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mock MongoDB client."""
        mock_client = Mock()
        mock_client.connected = True

        # Mock collection
        mock_collection = Mock()
        mock_collection.replace_one = Mock(return_value=Mock(upserted_id="exp1"))
        mock_client.get_collection = Mock(return_value=mock_collection)
        return mock_client

    @pytest.fixture
    def storage(self, mock_mongo_client):
        """Create an ExperimentStorage instance."""
        return ExperimentStorage(mock_mongo_client)

    @pytest.mark.unit
    def test_storage_creation(self, mock_mongo_client):
        """Test creating ExperimentStorage."""
        storage = ExperimentStorage(mock_mongo_client)

        assert storage.mongo_client is not None
        assert storage.collection_name == "virtual_experiments"

    @pytest.mark.unit
    def test_store_experiment_result(self, storage):
        """Test storing experiment result."""
        result = ExperimentResult(
            experiment_id="exp1",
            model_id="model1",
            design_data={"design_type": "factorial"},
            results={"quality": 0.8},
        )

        doc_id = storage.store_experiment_result(result)

        assert doc_id == "exp1"
        storage.mongo_client.get_collection.assert_called()

    @pytest.mark.unit
    def test_store_experiment_result_not_connected(self):
        """Test storing when MongoDB is not connected."""
        mock_client = Mock()
        mock_client.connected = False
        storage = ExperimentStorage(mock_client)

        result = ExperimentResult(experiment_id="exp1", model_id="model1", design_data={}, results={})

        with pytest.raises(ConnectionError):
            storage.store_experiment_result(result)

    @pytest.mark.unit
    def test_store_experiment_design(self, storage):
        """Test storing experiment design."""
        design_data = {
            "design_type": "factorial",
            "num_samples": 100,
            "parameter_ranges": {"power": (100, 300)},
        }

        doc_id = storage.store_experiment_design("exp1", design_data)

        assert doc_id == "exp1"
        storage.mongo_client.get_collection.assert_called()

    @pytest.mark.unit
    def test_store_comparison_results(self, storage):
        """Test storing comparison results."""
        comparison_data = {"metric1": 0.5, "metric2": 0.8, "agreement_score": 0.9}

        doc_id = storage.store_comparison_results(experiment_id="exp1", model_id="model1", comparison_data=comparison_data)

        assert doc_id == "exp1_model1"
        storage.mongo_client.get_collection.assert_called()

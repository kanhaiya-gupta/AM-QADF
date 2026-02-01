"""
Unit tests for sensitivity analysis query.

Tests for SensitivityQuery class.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from datetime import datetime
from am_qadf.analytics.sensitivity_analysis.query import SensitivityQuery


class TestSensitivityQuery:
    """Test suite for SensitivityQuery class."""

    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mock MongoDB client."""
        client = Mock()
        client.connected = True
        client.get_collection = Mock(return_value=Mock())
        return client

    @pytest.fixture
    def query_client(self, mock_mongo_client):
        """Create a SensitivityQuery instance."""
        return SensitivityQuery(mock_mongo_client)

    @pytest.mark.unit
    def test_query_creation(self, mock_mongo_client):
        """Test creating SensitivityQuery."""
        query = SensitivityQuery(mock_mongo_client)

        assert query.mongo_client == mock_mongo_client
        assert query.collection_name == "sensitivity_results"

    @pytest.mark.unit
    def test_query_sensitivity_results_all(self, query_client, mock_mongo_client):
        """Test querying all sensitivity results."""
        mock_collection = Mock()
        mock_collection.find.return_value = [
            {
                "_id": "result1",
                "model_id": "model1",
                "method": "sobol",
                "parameter_names": ["param1"],
                "sensitivity_indices": {"S1_param1": 0.5},
            },
            {
                "_id": "result2",
                "model_id": "model2",
                "method": "morris",
                "parameter_names": ["param2"],
                "sensitivity_indices": {"mu_star_param2": 0.3},
            },
        ]
        mock_mongo_client.get_collection.return_value = mock_collection

        results = query_client.query_sensitivity_results()

        assert len(results) == 2
        assert results[0]["_id"] == "result1"
        assert results[1]["_id"] == "result2"

    @pytest.mark.unit
    def test_query_sensitivity_results_by_model_id(self, query_client, mock_mongo_client):
        """Test querying sensitivity results by model ID."""
        mock_collection = Mock()
        mock_collection.find.return_value = [{"_id": "result1", "model_id": "model1", "method": "sobol"}]
        mock_mongo_client.get_collection.return_value = mock_collection

        results = query_client.query_sensitivity_results(model_id="model1")

        assert len(results) == 1
        assert results[0]["model_id"] == "model1"
        mock_collection.find.assert_called_once_with({"model_id": "model1"})

    @pytest.mark.unit
    def test_query_sensitivity_results_by_method(self, query_client, mock_mongo_client):
        """Test querying sensitivity results by method."""
        mock_collection = Mock()
        mock_collection.find.return_value = []
        mock_mongo_client.get_collection.return_value = mock_collection

        results = query_client.query_sensitivity_results(method="sobol")

        assert isinstance(results, list)
        mock_collection.find.assert_called_once_with({"method": "sobol"})

    @pytest.mark.unit
    def test_query_sensitivity_results_by_variable(self, query_client, mock_mongo_client):
        """Test querying sensitivity results by variable."""
        mock_collection = Mock()
        mock_collection.find.return_value = []
        mock_mongo_client.get_collection.return_value = mock_collection

        results = query_client.query_sensitivity_results(variable="param1")

        assert isinstance(results, list)
        # Check that query includes variable filter
        call_args = mock_collection.find.call_args[0][0]
        assert "parameter_names" in call_args

    @pytest.mark.unit
    def test_query_sensitivity_results_by_analysis_id(self, query_client, mock_mongo_client):
        """Test querying sensitivity results by analysis ID."""
        mock_collection = Mock()
        mock_collection.find.return_value = [{"_id": "analysis1", "model_id": "model1"}]
        mock_mongo_client.get_collection.return_value = mock_collection

        results = query_client.query_sensitivity_results(analysis_id="analysis1")

        assert len(results) == 1
        mock_collection.find.assert_called_once_with({"_id": "analysis1"})

    @pytest.mark.unit
    def test_query_sensitivity_results_not_connected(self, mock_mongo_client):
        """Test querying when MongoDB client is not connected."""
        mock_mongo_client.connected = False
        query = SensitivityQuery(mock_mongo_client)

        with pytest.raises(ConnectionError, match="MongoDB client not connected"):
            query.query_sensitivity_results()

    @pytest.mark.unit
    def test_compare_sensitivity(self, query_client, mock_mongo_client):
        """Test comparing sensitivity across multiple models."""
        mock_collection = Mock()
        mock_collection.find.return_value = [
            {
                "_id": "result1",
                "model_id": "model1",
                "method": "sobol",
                "sensitivity_indices": {"S1_param1": 0.5, "S1_param2": 0.3},
                "timestamp": "2023-01-01T00:00:00",
            },
            {
                "_id": "result2",
                "model_id": "model2",
                "method": "sobol",
                "sensitivity_indices": {"S1_param1": 0.6, "S1_param2": 0.4},
                "timestamp": "2023-01-02T00:00:00",
            },
        ]
        mock_mongo_client.get_collection.return_value = mock_collection

        df = query_client.compare_sensitivity(["model1", "model2"], method="sobol")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "model_id" in df.columns

    @pytest.mark.unit
    def test_compare_sensitivity_empty_results(self, query_client, mock_mongo_client):
        """Test comparing sensitivity when no results are found."""
        mock_collection = Mock()
        mock_collection.find.return_value = []
        mock_mongo_client.get_collection.return_value = mock_collection

        df = query_client.compare_sensitivity(["model1"], method="sobol")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @pytest.mark.unit
    def test_analyze_sensitivity_trends(self, query_client, mock_mongo_client):
        """Test analyzing sensitivity trends over time."""
        mock_collection = Mock()
        mock_collection.find.return_value = [
            {
                "_id": "result1",
                "model_id": "model1",
                "method": "sobol",
                "sensitivity_indices": {"S1_param1": 0.5},
                "timestamp": "2023-01-01T00:00:00",
            },
            {
                "_id": "result2",
                "model_id": "model1",
                "method": "sobol",
                "sensitivity_indices": {"S1_param1": 0.6},
                "timestamp": "2023-01-02T00:00:00",
            },
        ]
        mock_mongo_client.get_collection.return_value = mock_collection

        trends = query_client.analyze_sensitivity_trends("model1")

        assert isinstance(trends, dict)
        assert "sobol" in trends
        assert "S1_param1" in trends["sobol"]
        assert "mean" in trends["sobol"]["S1_param1"]
        assert "std" in trends["sobol"]["S1_param1"]

    @pytest.mark.unit
    def test_analyze_sensitivity_trends_with_time_range(self, query_client, mock_mongo_client):
        """Test analyzing sensitivity trends with time range."""
        mock_collection = Mock()
        mock_collection.find.return_value = [
            {
                "_id": "result1",
                "model_id": "model1",
                "method": "sobol",
                "sensitivity_indices": {"S1_param1": 0.5},
                "timestamp": "2023-01-15T00:00:00",
            }
        ]
        mock_mongo_client.get_collection.return_value = mock_collection

        time_range = (datetime(2023, 1, 1), datetime(2023, 1, 31))
        trends = query_client.analyze_sensitivity_trends("model1", time_range=time_range)

        assert isinstance(trends, dict)

    @pytest.mark.unit
    def test_analyze_sensitivity_trends_no_results(self, query_client, mock_mongo_client):
        """Test analyzing sensitivity trends when no results are found."""
        mock_collection = Mock()
        mock_collection.find.return_value = []
        mock_mongo_client.get_collection.return_value = mock_collection

        trends = query_client.analyze_sensitivity_trends("model1")

        assert isinstance(trends, dict)
        assert len(trends) == 0

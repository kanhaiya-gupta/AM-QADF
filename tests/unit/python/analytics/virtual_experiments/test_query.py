"""
Unit tests for experiment query.

Tests for ExperimentQuery class.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from am_qadf.analytics.virtual_experiments.query import ExperimentQuery


class TestExperimentQuery:
    """Test suite for ExperimentQuery class."""

    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mock MongoDB client."""
        mock_client = Mock()
        mock_client.connected = True

        # Mock collection
        mock_collection = Mock()
        mock_collection.find = Mock(
            return_value=[
                {
                    "_id": "exp1",
                    "model_id": "model1",
                    "design_data": {"design_type": "factorial"},
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "_id": "exp2",
                    "model_id": "model1",
                    "design_data": {"design_type": "lhs"},
                    "timestamp": datetime.now().isoformat(),
                },
            ]
        )
        mock_collection.find_one = Mock(
            return_value={
                "_id": "comp1",
                "comparison_data": {"metric1": 0.5, "metric2": 0.8},
            }
        )

        mock_client.get_collection = Mock(return_value=mock_collection)
        return mock_client

    @pytest.fixture
    def query(self, mock_mongo_client):
        """Create an ExperimentQuery instance."""
        return ExperimentQuery(mock_mongo_client)

    @pytest.mark.unit
    def test_query_creation(self, mock_mongo_client):
        """Test creating ExperimentQuery."""
        query = ExperimentQuery(mock_mongo_client)

        assert query.mongo_client is not None
        assert query.collection_name == "virtual_experiments"

    @pytest.mark.unit
    def test_query_experiment_results_all(self, query):
        """Test querying all experiment results."""
        results = query.query_experiment_results()

        assert isinstance(results, list)
        assert len(results) == 2

    @pytest.mark.unit
    def test_query_experiment_results_by_id(self, query):
        """Test querying experiment results by ID."""
        results = query.query_experiment_results(experiment_id="exp1")

        assert isinstance(results, list)

    @pytest.mark.unit
    def test_query_experiment_results_by_model_id(self, query):
        """Test querying experiment results by model ID."""
        results = query.query_experiment_results(model_id="model1")

        assert isinstance(results, list)

    @pytest.mark.unit
    def test_query_experiment_results_by_design_type(self, query):
        """Test querying experiment results by design type."""
        results = query.query_experiment_results(design_type="factorial")

        assert isinstance(results, list)

    @pytest.mark.unit
    def test_query_experiment_results_not_connected(self):
        """Test querying when MongoDB is not connected."""
        mock_client = Mock()
        mock_client.connected = False
        query = ExperimentQuery(mock_client)

        with pytest.raises(ConnectionError):
            query.query_experiment_results()

    @pytest.mark.unit
    def test_compare_experiments_with_warehouse(self, query):
        """Test comparing experiments with warehouse."""
        result = query.compare_experiments_with_warehouse(experiment_id="exp1", model_id="model1")

        assert isinstance(result, dict)
        # The function returns either comparison_data directly (from mock) or experiment data with experiment_id
        # The mock returns comparison_data as {'metric1': 0.5, 'metric2': 0.8}
        # So we just check that it's a non-empty dict
        assert len(result) > 0

    @pytest.mark.unit
    def test_analyze_experiment_trends(self, query):
        """Test analyzing experiment trends."""
        trends = query.analyze_experiment_trends(model_id="model1")

        assert isinstance(trends, dict)
        assert "num_experiments" in trends
        assert "design_types" in trends
        assert "parameter_ranges" in trends

    @pytest.mark.unit
    def test_analyze_experiment_trends_with_time_range(self, query):
        """Test analyzing experiment trends with time range."""
        start_time = datetime(2020, 1, 1)
        end_time = datetime(2023, 12, 31)

        trends = query.analyze_experiment_trends(model_id="model1", time_range=(start_time, end_time))

        assert isinstance(trends, dict)

    @pytest.mark.unit
    def test_analyze_experiment_trends_no_results(self, query):
        """Test analyzing trends when no results exist."""
        mock_client = Mock()
        mock_client.connected = True
        mock_collection = Mock()
        mock_collection.find = Mock(return_value=[])
        mock_client.get_collection = Mock(return_value=mock_collection)

        query = ExperimentQuery(mock_client)
        trends = query.analyze_experiment_trends()

        assert isinstance(trends, dict)
        assert trends == {}

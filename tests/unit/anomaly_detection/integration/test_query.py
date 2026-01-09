"""
Unit tests for anomaly query.

Tests for AnomalyQuery class.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from am_qadf.anomaly_detection.integration.query import AnomalyQuery


class TestAnomalyQuery:
    """Test suite for AnomalyQuery class."""

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
                    "_id": "det1",
                    "model_id": "model1",
                    "methods": ["isolation_forest"],
                    "anomalies": {
                        "isolation_forest": [
                            {
                                "is_anomaly": True,
                                "anomaly_score": 0.9,
                                "anomaly_type": "spatial",
                            },
                            {
                                "is_anomaly": False,
                                "anomaly_score": 0.2,
                                "anomaly_type": "spatial",
                            },
                        ]
                    },
                    "timestamp": datetime.now().isoformat(),
                    "total_points": 100,
                },
                {
                    "_id": "det2",
                    "model_id": "model2",
                    "methods": ["dbscan"],
                    "anomalies": {
                        "dbscan": [
                            {
                                "is_anomaly": True,
                                "anomaly_score": 0.6,
                                "anomaly_type": "temporal",
                            }
                        ]
                    },
                    "timestamp": datetime.now().isoformat(),
                    "total_points": 200,
                },
            ]
        )

        mock_client.get_collection = Mock(return_value=mock_collection)
        return mock_client

    @pytest.fixture
    def query(self, mock_mongo_client):
        """Create an AnomalyQuery instance."""
        return AnomalyQuery(mock_mongo_client)

    @pytest.mark.unit
    def test_query_creation(self, mock_mongo_client):
        """Test creating AnomalyQuery."""
        query = AnomalyQuery(mock_mongo_client)

        assert query.mongo_client is not None
        assert query.collection_name == "anomaly_detections"

    @pytest.mark.unit
    def test_query_anomalies_all(self, query):
        """Test querying all anomalies."""
        results = query.query_anomalies()

        assert isinstance(results, list)
        assert len(results) == 2

    @pytest.mark.unit
    def test_query_anomalies_by_model_id(self, query):
        """Test querying anomalies by model ID."""
        # Mock collection with filtered results
        mock_collection = Mock()
        mock_collection.find = Mock(
            return_value=[
                {
                    "_id": "det1",
                    "model_id": "model1",
                    "methods": ["isolation_forest"],
                    "anomalies": {},
                }
            ]
        )
        query.mongo_client.get_collection = Mock(return_value=mock_collection)

        results = query.query_anomalies(model_id="model1")

        assert isinstance(results, list)

    @pytest.mark.unit
    def test_query_anomalies_by_method(self, query):
        """Test querying anomalies by method."""
        results = query.query_anomalies(method="isolation_forest")

        assert isinstance(results, list)

    @pytest.mark.unit
    def test_query_anomalies_by_anomaly_type(self, query):
        """Test querying anomalies by anomaly type."""
        results = query.query_anomalies(anomaly_type="spatial")

        assert isinstance(results, list)
        # Should filter by anomaly type
        if len(results) > 0:
            assert "anomaly" in results[0] or "anomalies" in results[0]

    @pytest.mark.unit
    def test_query_anomalies_by_severity(self, query):
        """Test querying anomalies by severity."""
        results = query.query_anomalies(severity="high")

        assert isinstance(results, list)
        # Should filter by severity (high = score >= 0.8)
        if len(results) > 0:
            assert "anomaly" in results[0] or "anomalies" in results[0]

    @pytest.mark.unit
    def test_query_anomalies_by_detection_id(self, query):
        """Test querying anomalies by detection ID."""
        # Mock collection with specific ID
        mock_collection = Mock()
        mock_collection.find = Mock(
            return_value=[
                {
                    "_id": "det1",
                    "model_id": "model1",
                    "methods": ["isolation_forest"],
                    "anomalies": {},
                }
            ]
        )
        query.mongo_client.get_collection = Mock(return_value=mock_collection)

        results = query.query_anomalies(detection_id="det1")

        assert isinstance(results, list)

    @pytest.mark.unit
    def test_query_anomalies_not_connected(self):
        """Test querying when MongoDB is not connected."""
        mock_client = Mock()
        mock_client.connected = False
        query = AnomalyQuery(mock_client)

        with pytest.raises(ConnectionError):
            query.query_anomalies()

    @pytest.mark.unit
    def test_query_anomaly_results(self, query):
        """Test querying anomaly results in formatted format."""
        # Mock collection with results structure
        mock_collection = Mock()
        mock_collection.find = Mock(
            return_value=[
                {
                    "_id": "det1",
                    "model_id": "model1",
                    "methods": ["isolation_forest"],
                    "results": {
                        "isolation_forest": {
                            "num_anomalies": 5,
                            "anomaly_labels": [1, 0, 1, 0, 1],
                        }
                    },
                    "total_points": 100,
                    "timestamp": datetime.now(),
                }
            ]
        )
        query.mongo_client.get_collection = Mock(return_value=mock_collection)

        results = query.query_anomaly_results(model_id="model1")

        assert isinstance(results, list)
        if len(results) > 0:
            assert "detection_id" in results[0]
            assert "model_id" in results[0]
            assert "method" in results[0]
            assert "num_anomalies" in results[0]
            assert "anomaly_rate" in results[0]

    @pytest.mark.unit
    def test_query_anomaly_results_with_limit(self, query):
        """Test querying anomaly results with limit."""
        # Mock collection
        mock_collection = Mock()
        mock_collection.find = Mock(
            return_value=[
                {
                    "_id": f"det{i}",
                    "model_id": "model1",
                    "methods": ["isolation_forest"],
                    "anomalies": {},
                }
                for i in range(5)
            ]
        )
        query.mongo_client.get_collection = Mock(return_value=mock_collection)

        results = query.query_anomaly_results(limit=2)

        assert isinstance(results, list)
        assert len(results) <= 2

    @pytest.mark.unit
    def test_analyze_anomaly_trends(self, query):
        """Test analyzing anomaly trends."""
        # Mock collection
        mock_collection = Mock()
        mock_collection.find = Mock(
            return_value=[
                {
                    "_id": "det1",
                    "model_id": "model1",
                    "methods": ["isolation_forest"],
                    "anomalies": {"isolation_forest": [{"is_anomaly": True, "anomaly_score": 0.9}]},
                    "timestamp": datetime.now().isoformat(),
                }
            ]
        )
        query.mongo_client.get_collection = Mock(return_value=mock_collection)

        trends = query.analyze_anomaly_trends(model_ids=["model1", "model2"])

        assert isinstance(trends, dict)
        assert "total_detections" in trends
        assert "models_analyzed" in trends
        assert "methods_used" in trends
        assert "anomaly_counts_by_method" in trends

    @pytest.mark.unit
    def test_analyze_anomaly_trends_with_time_range(self, query):
        """Test analyzing anomaly trends with time range."""
        start_time = datetime(2020, 1, 1)
        end_time = datetime(2023, 12, 31)

        # Mock collection
        mock_collection = Mock()
        mock_collection.find = Mock(
            return_value=[
                {
                    "_id": "det1",
                    "model_id": "model1",
                    "methods": ["isolation_forest"],
                    "anomalies": {},
                    "timestamp": datetime(2022, 6, 1).isoformat(),
                }
            ]
        )
        query.mongo_client.get_collection = Mock(return_value=mock_collection)

        trends = query.analyze_anomaly_trends(model_ids=["model1"], time_range=(start_time, end_time))

        assert isinstance(trends, dict)

    @pytest.mark.unit
    def test_query_anomaly_patterns(self, query):
        """Test querying anomaly patterns."""
        # Mock patterns collection
        patterns_collection = Mock()
        patterns_collection.find = Mock(
            return_value=[
                {
                    "_id": "pattern1",
                    "model_id": "model1",
                    "patterns": {"pattern_type": "spatial_cluster"},
                }
            ]
        )
        query.mongo_client.get_collection = Mock(return_value=patterns_collection)

        patterns = query.query_anomaly_patterns(model_id="model1")

        assert isinstance(patterns, list)
        if len(patterns) > 0:
            assert "model_id" in patterns[0]
            assert "patterns" in patterns[0]

    @pytest.mark.unit
    def test_query_anomaly_patterns_by_type(self, query):
        """Test querying anomaly patterns by pattern type."""
        patterns_collection = Mock()
        patterns_collection.find = Mock(
            return_value=[
                {
                    "_id": "pattern1",
                    "model_id": "model1",
                    "patterns": {"pattern_type": "spatial_cluster"},
                }
            ]
        )
        query.mongo_client.get_collection = Mock(return_value=patterns_collection)

        patterns = query.query_anomaly_patterns(model_id="model1", pattern_type="spatial_cluster")

        assert isinstance(patterns, list)

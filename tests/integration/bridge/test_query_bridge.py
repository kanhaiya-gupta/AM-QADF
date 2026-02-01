"""
Bridge tests: query (QueryResult, MongoDBQueryClient, PointConverter) via am_qadf_native.

Aligned with docs/Tests/Test_New_Plans.md (tests/integration/bridge/).
"""

import pytest


@pytest.mark.integration
@pytest.mark.bridge
class TestQueryBridge:
    """Python → C++ query API."""

    def test_query_result_create_and_num_points(self, native_module):
        """QueryResult: set points/values, num_points and empty."""
        QueryResult = native_module.QueryResult
        r = QueryResult()
        r.points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        r.values = [10.0, 20.0]
        r.timestamps = [0.0, 1.0]
        r.layers = [0, 0]
        assert r.num_points() == 2
        assert r.empty() is False

    def test_query_result_empty(self, native_module):
        """QueryResult.empty() when no points."""
        QueryResult = native_module.QueryResult
        r = QueryResult()
        assert r.empty() is True

    def test_mongodb_query_client_construct(self, native_module):
        """MongoDBQueryClient can be constructed (no server required)."""
        MongoDBQueryClient = native_module.MongoDBQueryClient
        client = MongoDBQueryClient("mongodb://localhost:27017", "test_db")
        assert client is not None

    def test_point_converter_points_to_array(self, native_module):
        """points_to_eigen_matrix converts list of [x,y,z] to array (Eigen/numpy)."""
        if not hasattr(native_module, "points_to_eigen_matrix"):
            pytest.skip("points_to_eigen_matrix not exposed (Eigen not available)")
        points = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        arr = native_module.points_to_eigen_matrix(points)
        assert arr is not None
        assert arr.shape[0] == 2
        assert arr.shape[1] == 3

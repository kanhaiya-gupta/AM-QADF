"""
Integration tests for SPC storage.

Tests SPCStorage with MongoDB integration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime

try:
    from am_qadf.analytics.spc.spc_storage import SPCStorage
    from am_qadf.analytics.spc.baseline_calculation import BaselineStatistics
    from am_qadf.analytics.spc.control_charts import ControlChartResult
    from am_qadf.analytics.spc.process_capability import ProcessCapabilityResult
    from am_qadf.analytics.spc.multivariate_spc import MultivariateSPCResult
except ImportError:
    pytest.skip("SPC storage module not available", allow_module_level=True)


class TestSPCStorage:
    """Integration tests for SPCStorage."""

    @pytest.fixture
    def mock_mongo_client(self):
        """Create mock MongoDB client."""
        mock_client = Mock()
        mock_client.connected = True
        mock_client.is_connected = Mock(return_value=True)
        mock_client.get_collection = Mock(return_value=Mock())
        return mock_client

    @pytest.fixture
    def storage(self, mock_mongo_client):
        """Create SPCStorage instance with mock MongoDB."""
        return SPCStorage(mock_mongo_client)

    @pytest.fixture
    def baseline(self):
        """Create sample baseline statistics."""
        return BaselineStatistics(
            mean=10.0,
            std=2.0,
            median=10.0,
            min=5.0,
            max=15.0,
            range=10.0,
            sample_size=100,
            subgroup_size=5,
            within_subgroup_std=1.8,
            between_subgroup_std=0.6,
            overall_std=2.0,
            calculated_at=datetime.now(),
            metadata={"method": "standard"},
        )

    @pytest.fixture
    def chart_result(self):
        """Create sample control chart result."""
        return ControlChartResult(
            chart_type="xbar",
            center_line=10.0,
            upper_control_limit=13.0,
            lower_control_limit=7.0,
            sample_values=np.array([10.0, 11.0, 9.0, 12.0, 8.0]),
            sample_indices=np.array([0, 1, 2, 3, 4]),
            out_of_control_points=[3],
            baseline_stats={"mean": 10.0, "std": 1.0},
            metadata={"subgroup_size": 5},
        )

    @pytest.fixture
    def capability_result(self):
        """Create sample process capability result."""
        return ProcessCapabilityResult(
            cp=1.5,
            cpk=1.3,
            pp=1.4,
            ppk=1.2,
            cpu=1.5,
            cpl=1.3,
            specification_limits=(12.0, 8.0),
            target_value=10.0,
            process_mean=10.1,
            process_std=1.0,
            within_subgroup_std=0.95,
            overall_std=1.05,
            capability_rating="Adequate",
            metadata={"sample_size": 100},
        )

    @pytest.fixture
    def multivariate_result(self):
        """Create sample multivariate SPC result."""
        return MultivariateSPCResult(
            hotelling_t2=np.array([1.0, 2.0, 3.0, 15.0, 2.5]),  # One OOC point
            ucl_t2=10.0,
            control_limits={"variable_0": {"mean": 0.0, "ucl": 3.0, "lcl": -3.0}},
            out_of_control_points=[3],
            principal_components=None,
            explained_variance=None,
            contribution_analysis={3: ["variable_0", "variable_1"]},
            baseline_mean=np.array([0.0, 0.0]),
            baseline_covariance=np.eye(2),
            metadata={"n_variables": 2},
        )

    @pytest.mark.integration
    def test_storage_creation(self, storage):
        """Test creating SPCStorage."""
        assert storage is not None
        assert storage.mongo_client is not None

    @pytest.mark.integration
    def test_check_connection(self, storage, mock_mongo_client):
        """Test connection check."""
        assert storage._check_connection() == True

        # Test with disconnected client
        mock_mongo_client.connected = False
        # Also need to update is_connected method if it exists
        if hasattr(mock_mongo_client, "is_connected"):
            mock_mongo_client.is_connected = Mock(return_value=False)
        assert storage._check_connection() == False

    @pytest.mark.integration
    def test_save_baseline(self, storage, baseline, mock_mongo_client):
        """Test saving baseline statistics."""
        collection_mock = Mock()
        mock_mongo_client.get_collection.return_value = collection_mock

        baseline_id = storage.save_baseline(
            model_id="test_model", signal_name="test_signal", baseline=baseline, metadata={"test": "metadata"}
        )

        assert isinstance(baseline_id, str)
        assert "test_model" in baseline_id
        assert "test_signal" in baseline_id
        # Verify replace_one was called
        assert collection_mock.replace_one.called

    @pytest.mark.integration
    def test_load_baseline(self, storage, baseline, mock_mongo_client):
        """Test loading baseline statistics."""
        collection_mock = Mock()
        mock_mongo_client.get_collection.return_value = collection_mock

        # Mock document from database
        mock_doc = {
            "_id": "test_model_test_signal_2026-01-01T00:00:00",
            "model_id": "test_model",
            "signal_name": "test_signal",
            "baseline": {
                "mean": 10.0,
                "std": 2.0,
                "median": 10.0,
                "min": 5.0,
                "max": 15.0,
                "range": 10.0,
                "sample_size": 100,
                "subgroup_size": 5,
                "within_subgroup_std": 1.8,
                "between_subgroup_std": 0.6,
                "overall_std": 2.0,
                "calculated_at": datetime.now().isoformat(),
                "metadata": {},
            },
            "created_at": datetime.now().isoformat(),
        }
        collection_mock.find_one.return_value = mock_doc

        loaded_baseline = storage.load_baseline("test_model", "test_signal")

        assert isinstance(loaded_baseline, BaselineStatistics)
        assert loaded_baseline.mean == 10.0
        assert loaded_baseline.std == 2.0
        assert loaded_baseline.sample_size == 100

    @pytest.mark.integration
    def test_load_baseline_not_found(self, storage, mock_mongo_client):
        """Test loading baseline that doesn't exist."""
        collection_mock = Mock()
        mock_mongo_client.get_collection.return_value = collection_mock
        collection_mock.find_one.return_value = None

        with pytest.raises(ValueError, match="Baseline not found"):
            storage.load_baseline("test_model", "test_signal")

    @pytest.mark.integration
    def test_save_control_chart(self, storage, chart_result, mock_mongo_client):
        """Test saving control chart result."""
        collection_mock = Mock()
        mock_mongo_client.get_collection.return_value = collection_mock

        chart_id = storage.save_control_chart(
            model_id="test_model", chart_result=chart_result, metadata={"analysis_id": "test_analysis"}
        )

        assert isinstance(chart_id, str)
        assert collection_mock.replace_one.called

    @pytest.mark.integration
    def test_load_control_chart(self, storage, chart_result, mock_mongo_client):
        """Test loading control chart result."""
        collection_mock = Mock()
        mock_mongo_client.get_collection.return_value = collection_mock

        # Mock document
        mock_doc = {
            "_id": "test_chart_id",
            "model_id": "test_model",
            "chart_result": {
                "chart_type": "xbar",
                "center_line": 10.0,
                "upper_control_limit": 13.0,
                "lower_control_limit": 7.0,
                "upper_warning_limit": 12.0,
                "lower_warning_limit": 8.0,
                "sample_values": [10.0, 11.0, 9.0, 12.0, 8.0],
                "sample_indices": [0, 1, 2, 3, 4],
                "out_of_control_points": [3],
                "rule_violations": {},
                "baseline_stats": {"mean": 10.0, "std": 1.0},
                "metadata": {"subgroup_size": 5},
            },
            "created_at": datetime.now().isoformat(),
        }
        collection_mock.find_one.return_value = mock_doc

        loaded_chart = storage.load_control_chart("test_model", "test_chart_id")

        assert isinstance(loaded_chart, ControlChartResult)
        assert loaded_chart.chart_type == "xbar"
        assert loaded_chart.center_line == 10.0

    @pytest.mark.integration
    def test_save_capability_result(self, storage, capability_result, mock_mongo_client):
        """Test saving process capability result."""
        collection_mock = Mock()
        mock_mongo_client.get_collection.return_value = collection_mock

        capability_id = storage.save_capability_result(model_id="test_model", capability_result=capability_result)

        assert isinstance(capability_id, str)
        assert collection_mock.replace_one.called

    @pytest.mark.integration
    def test_save_multivariate_result(self, storage, multivariate_result, mock_mongo_client):
        """Test saving multivariate SPC result."""
        collection_mock = Mock()
        mock_mongo_client.get_collection.return_value = collection_mock

        result_id = storage.save_multivariate_result(model_id="test_model", multivariate_result=multivariate_result)

        assert isinstance(result_id, str)
        assert collection_mock.replace_one.called

    @pytest.mark.integration
    def test_query_spc_history(self, storage, mock_mongo_client):
        """Test querying SPC history."""
        collection_mock = Mock()
        mock_mongo_client.get_collection.return_value = collection_mock

        # Mock query results
        mock_docs = [
            {
                "_id": "baseline_1",
                "model_id": "test_model",
                "signal_name": "test_signal",
                "created_at": datetime.now().isoformat(),
                "result_type": "baselines",
            },
            {
                "_id": "chart_1",
                "model_id": "test_model",
                "created_at": datetime.now().isoformat(),
                "result_type": "control_charts",
            },
        ]
        # Mock cursor that supports sort() method
        mock_cursor = Mock()
        mock_cursor.sort.return_value = mock_docs
        collection_mock.find.return_value = mock_cursor

        results = storage.query_spc_history(model_id="test_model", signal_name="test_signal")

        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.integration
    def test_query_spc_history_with_time_range(self, storage, mock_mongo_client):
        """Test querying SPC history with time range."""
        collection_mock = Mock()
        mock_mongo_client.get_collection.return_value = collection_mock
        mock_cursor = Mock()
        mock_cursor.sort.return_value = []
        collection_mock.find.return_value = mock_cursor

        start_time = datetime(2026, 1, 1)
        end_time = datetime(2026, 1, 31)

        results = storage.query_spc_history(model_id="test_model", start_time=start_time, end_time=end_time)

        assert isinstance(results, list)
        # Verify query was constructed correctly
        assert collection_mock.find.called

    @pytest.mark.integration
    def test_query_spc_history_by_type(self, storage, mock_mongo_client):
        """Test querying SPC history filtered by result type."""
        collection_mock = Mock()
        mock_mongo_client.get_collection.return_value = collection_mock
        mock_cursor = Mock()
        mock_cursor.sort.return_value = []
        collection_mock.find.return_value = mock_cursor

        results = storage.query_spc_history(model_id="test_model", result_type="baseline")

        assert isinstance(results, list)

    @pytest.mark.integration
    def test_storage_disconnected_client(self):
        """Test storage operations with disconnected client."""
        mock_client = Mock()
        # Set both connected attribute and is_connected method to False
        mock_client.connected = False
        mock_client.is_connected = Mock(return_value=False)
        storage = SPCStorage(mock_client)

        baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=1
        )

        with pytest.raises(ConnectionError, match="MongoDB client not connected"):
            storage.save_baseline("test_model", "test_signal", baseline)

    @pytest.mark.integration
    def test_baseline_to_dict(self, storage, baseline):
        """Test converting baseline to dictionary."""
        baseline_dict = storage._baseline_to_dict(baseline)

        assert isinstance(baseline_dict, dict)
        assert baseline_dict["mean"] == 10.0
        assert baseline_dict["std"] == 2.0
        assert "calculated_at" in baseline_dict

    @pytest.mark.integration
    def test_control_chart_to_dict(self, storage, chart_result):
        """Test converting control chart result to dictionary."""
        chart_dict = storage._control_chart_to_dict(chart_result)

        assert isinstance(chart_dict, dict)
        assert chart_dict["chart_type"] == "xbar"
        assert chart_dict["center_line"] == 10.0
        assert isinstance(chart_dict["sample_values"], list)
        assert isinstance(chart_dict["sample_indices"], list)

    @pytest.mark.integration
    def test_multivariate_result_to_dict(self, storage, multivariate_result):
        """Test converting multivariate result to dictionary."""
        result_dict = storage._multivariate_result_to_dict(multivariate_result)

        assert isinstance(result_dict, dict)
        assert isinstance(result_dict["hotelling_t2"], list)
        assert result_dict["ucl_t2"] == 10.0
        assert isinstance(result_dict["baseline_mean"], list)
        assert isinstance(result_dict["baseline_covariance"], list)

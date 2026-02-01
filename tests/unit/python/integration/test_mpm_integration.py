"""
Unit tests for MPM integration.

Tests for MPMClient and MPMProcessData.
"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from am_qadf.integration.mpm_integration import (
    MPMClient,
    MPMProcessData,
    MPMStatus,
)


class TestMPMProcessData:
    """Test suite for MPMProcessData dataclass."""

    @pytest.mark.unit
    def test_mpm_process_data_creation(self):
        """Test creating MPMProcessData."""
        timestamp = datetime.now()
        process_data = MPMProcessData(
            process_id="proc123",
            build_id="build456",
            material="titanium",
            parameters={"laser_power": 200.0, "speed": 100.0},
            status=MPMStatus.RUNNING.value,
            start_time=timestamp,
            metadata={"operator": "user1"},
        )

        assert process_data.process_id == "proc123"
        assert process_data.build_id == "build456"
        assert process_data.material == "titanium"
        assert process_data.parameters["laser_power"] == 200.0
        assert process_data.status == MPMStatus.RUNNING.value
        assert process_data.start_time == timestamp
        assert process_data.metadata["operator"] == "user1"

    @pytest.mark.unit
    def test_mpm_process_data_to_dict(self):
        """Test converting MPMProcessData to dictionary."""
        timestamp = datetime.now()
        process_data = MPMProcessData(
            process_id="proc123",
            build_id="build456",
            material="titanium",
            parameters={"laser_power": 200.0},
            status=MPMStatus.RUNNING.value,
            start_time=timestamp,
        )

        data_dict = process_data.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict["process_id"] == "proc123"
        assert data_dict["build_id"] == "build456"
        assert isinstance(data_dict["start_time"], str)

    @pytest.mark.unit
    def test_mpm_process_data_from_dict(self):
        """Test creating MPMProcessData from dictionary."""
        data = {
            "process_id": "proc123",
            "build_id": "build456",
            "material": "titanium",
            "parameters": {"laser_power": 200.0},
            "status": MPMStatus.RUNNING.value,
            "start_time": "2024-01-01T00:00:00",
            "end_time": None,
            "metadata": {},
        }

        process_data = MPMProcessData.from_dict(data)

        assert process_data.process_id == "proc123"
        assert isinstance(process_data.start_time, datetime)
        assert process_data.end_time is None


class TestMPMClient:
    """Test suite for MPMClient class."""

    @pytest.fixture
    def mpm_client(self):
        """Create an MPMClient instance."""
        return MPMClient(
            base_url="http://localhost:8000",
            api_key="test_api_key",
            timeout=30.0,
        )

    @pytest.mark.unit
    def test_mpm_client_creation(self, mpm_client):
        """Test creating MPMClient."""
        assert mpm_client.base_url == "http://localhost:8000"
        assert mpm_client.api_key == "test_api_key"
        assert mpm_client.timeout == 30.0
        assert mpm_client.retry_on_failure is True
        assert mpm_client.max_retries == 3

    @pytest.mark.unit
    def test_mpm_client_creation_without_api_key(self):
        """Test creating MPMClient without API key."""
        client = MPMClient(base_url="http://localhost:8000")
        assert client.api_key is None
        assert "Authorization" not in client.session.headers

    @pytest.mark.unit
    def test_mpm_client_creation_custom_settings(self):
        """Test creating MPMClient with custom settings."""
        client = MPMClient(
            base_url="http://example.com",
            timeout=60.0,
            verify_ssl=False,
            retry_on_failure=False,
            max_retries=5,
        )
        assert client.timeout == 60.0
        assert client.verify_ssl is False
        assert client.retry_on_failure is False
        assert client.max_retries == 5

    @pytest.mark.unit
    @patch("am_qadf.integration.mpm_integration.requests.Session")
    def test_get_process_data_success(self, mock_session_class, mpm_client):
        """Test getting process data successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "process_id": "proc123",
            "build_id": "build456",
            "material": "titanium",
            "parameters": {"laser_power": 200.0},
            "status": MPMStatus.RUNNING.value,
            "start_time": "2024-01-01T00:00:00",
            "end_time": None,
            "metadata": {},
        }
        mock_response.content = b"{}"

        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mpm_client.session = mock_session

        process_data = mpm_client.get_process_data("proc123")

        assert isinstance(process_data, MPMProcessData)
        assert process_data.process_id == "proc123"
        assert process_data.build_id == "build456"

    @pytest.mark.unit
    @patch("am_qadf.integration.mpm_integration.requests.Session")
    def test_get_process_data_not_found(self, mock_session_class, mpm_client):
        """Test getting process data when process not found."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)

        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mpm_client.session = mock_session

        with pytest.raises(ValueError, match="Process proc123 not found"):
            mpm_client.get_process_data("proc123")

    @pytest.mark.unit
    @patch("am_qadf.integration.mpm_integration.requests.Session")
    def test_update_process_status_success(self, mock_session_class, mpm_client):
        """Test updating process status successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.content = b"{}"

        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mpm_client.session = mock_session

        result = mpm_client.update_process_status("proc123", MPMStatus.COMPLETED.value)

        assert result is True
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[1]["method"] == "PUT"
        assert call_args[1]["json"]["status"] == MPMStatus.COMPLETED.value

    @pytest.mark.unit
    def test_update_process_status_invalid_status(self, mpm_client):
        """Test updating process status with invalid status."""
        with pytest.raises(ValueError, match="Invalid status"):
            mpm_client.update_process_status("proc123", "invalid_status")

    @pytest.mark.unit
    @patch("am_qadf.integration.mpm_integration.requests.Session")
    def test_submit_quality_results_success(self, mock_session_class, mpm_client):
        """Test submitting quality results successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.content = b"{}"

        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mpm_client.session = mock_session

        results = {
            "overall_score": 0.95,
            "quality_scores": {"data_quality": 0.9, "signal_quality": 1.0},
        }

        result = mpm_client.submit_quality_results("proc123", results)

        assert result is True
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["json"]["results"] == results

    @pytest.mark.unit
    @patch("am_qadf.integration.mpm_integration.requests.Session")
    def test_get_process_parameters_success(self, mock_session_class, mpm_client):
        """Test getting process parameters successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "parameters": {
                "laser_power": 200.0,
                "scan_speed": 100.0,
                "layer_thickness": 0.1,
            }
        }
        mock_response.content = b"{}"

        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mpm_client.session = mock_session

        parameters = mpm_client.get_process_parameters("proc123")

        assert isinstance(parameters, dict)
        assert parameters["laser_power"] == 200.0
        assert parameters["scan_speed"] == 100.0

    @pytest.mark.unit
    @patch("am_qadf.integration.mpm_integration.requests.Session")
    def test_list_processes_success(self, mock_session_class, mpm_client):
        """Test listing processes successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "processes": [
                {
                    "process_id": "proc1",
                    "build_id": "build1",
                    "material": "titanium",
                    "parameters": {},
                    "status": MPMStatus.RUNNING.value,
                    "start_time": "2024-01-01T00:00:00",
                    "end_time": None,
                    "metadata": {},
                },
                {
                    "process_id": "proc2",
                    "build_id": "build2",
                    "material": "aluminum",
                    "parameters": {},
                    "status": MPMStatus.COMPLETED.value,
                    "start_time": "2024-01-01T01:00:00",
                    "end_time": "2024-01-01T02:00:00",
                    "metadata": {},
                },
            ]
        }
        mock_response.content = b"{}"

        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mpm_client.session = mock_session

        processes = mpm_client.list_processes(limit=10, offset=0)

        assert len(processes) == 2
        assert isinstance(processes[0], MPMProcessData)
        assert processes[0].process_id == "proc1"
        assert processes[1].process_id == "proc2"

    @pytest.mark.unit
    @patch("am_qadf.integration.mpm_integration.requests.Session")
    def test_list_processes_with_filters(self, mock_session_class, mpm_client):
        """Test listing processes with filters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"processes": []}
        mock_response.content = b"{}"

        mock_session = Mock()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mpm_client.session = mock_session

        processes = mpm_client.list_processes(
            build_id="build123",
            status=MPMStatus.RUNNING.value,
            limit=50,
            offset=10,
        )

        assert isinstance(processes, list)
        call_args = mock_session.request.call_args
        assert call_args[1]["params"]["build_id"] == "build123"
        assert call_args[1]["params"]["status"] == MPMStatus.RUNNING.value

    @pytest.mark.unit
    @patch("am_qadf.integration.mpm_integration.requests.Session")
    def test_health_check_success(self, mock_session_class, mpm_client):
        """Test health check when system is healthy."""
        mock_response = Mock()
        mock_response.status_code = 200

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = {}
        mpm_client.session = mock_session

        is_healthy = mpm_client.health_check()

        assert is_healthy is True

    @pytest.mark.unit
    @patch("am_qadf.integration.mpm_integration.requests.Session")
    def test_health_check_failure(self, mock_session_class, mpm_client):
        """Test health check when system is unhealthy."""
        mock_session = Mock()
        mock_session.get.side_effect = requests.exceptions.RequestException()
        mock_session.headers = {}
        mpm_client.session = mock_session

        is_healthy = mpm_client.health_check()

        assert is_healthy is False

    @pytest.mark.unit
    @patch("am_qadf.integration.mpm_integration.requests.Session")
    def test_retry_on_failure(self, mock_session_class, mpm_client):
        """Test retry logic on failure."""
        mock_session = Mock()
        mock_session.request.side_effect = [
            requests.exceptions.ConnectionError(),
            requests.exceptions.ConnectionError(),
            Mock(status_code=200, json=lambda: {}, content=b"{}", raise_for_status=lambda: None),
        ]
        mock_session.headers = {}
        mpm_client.session = mock_session
        mpm_client.max_retries = 3

        # Should succeed after retries
        result = mpm_client._make_request("GET", "/api/v1/processes/proc123")
        assert mock_session.request.call_count == 3

    @pytest.mark.unit
    def test_close_session(self, mpm_client):
        """Test closing client session."""
        mock_session = Mock()
        mpm_client.session = mock_session

        mpm_client.close()

        mock_session.close.assert_called_once()

    @pytest.mark.unit
    def test_context_manager(self, mpm_client):
        """Test using MPMClient as context manager."""
        mock_session = Mock()
        mpm_client.session = mock_session

        with mpm_client:
            pass

        mock_session.close.assert_called_once()

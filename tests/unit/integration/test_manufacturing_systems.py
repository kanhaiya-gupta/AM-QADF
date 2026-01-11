"""
Unit tests for manufacturing systems integration.

Tests for EquipmentClient and EquipmentStatus.
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime

from am_qadf.integration.manufacturing_systems import (
    EquipmentClient,
    EquipmentStatus,
    EquipmentType,
    EquipmentStatusValue,
)


class TestEquipmentStatus:
    """Test suite for EquipmentStatus dataclass."""

    @pytest.mark.unit
    def test_equipment_status_creation(self):
        """Test creating EquipmentStatus."""
        timestamp = datetime.now()
        status = EquipmentStatus(
            equipment_id="equip1",
            status=EquipmentStatusValue.RUNNING.value,
            current_operation="milling",
            progress=0.75,
            error_message=None,
            timestamp=timestamp,
            metadata={"temperature": 100.0},
        )

        assert status.equipment_id == "equip1"
        assert status.status == EquipmentStatusValue.RUNNING.value
        assert status.current_operation == "milling"
        assert status.progress == 0.75
        assert status.error_message is None
        assert status.timestamp == timestamp
        assert status.metadata["temperature"] == 100.0

    @pytest.mark.unit
    def test_equipment_status_to_dict(self):
        """Test converting EquipmentStatus to dictionary."""
        status = EquipmentStatus(
            equipment_id="equip1",
            status=EquipmentStatusValue.IDLE.value,
            progress=0.0,
        )

        status_dict = status.to_dict()

        assert isinstance(status_dict, dict)
        assert status_dict["equipment_id"] == "equip1"
        assert isinstance(status_dict["timestamp"], str)

    @pytest.mark.unit
    def test_equipment_status_from_dict(self):
        """Test creating EquipmentStatus from dictionary."""
        data = {
            "equipment_id": "equip1",
            "status": EquipmentStatusValue.RUNNING.value,
            "current_operation": "milling",
            "progress": 0.5,
            "error_message": None,
            "timestamp": "2024-01-01T00:00:00",
            "metadata": {},
        }

        status = EquipmentStatus.from_dict(data)

        assert status.equipment_id == "equip1"
        assert isinstance(status.timestamp, datetime)


class TestEquipmentClient:
    """Test suite for EquipmentClient class."""

    @pytest.fixture
    def equipment_client(self):
        """Create an EquipmentClient instance."""
        connection_config = {
            "type": "network",
            "host": "localhost",
            "port": 8080,
        }
        return EquipmentClient(
            equipment_type=EquipmentType.CNC_MILL.value,
            equipment_id="cnc1",
            connection_config=connection_config,
        )

    @pytest.mark.unit
    def test_equipment_client_creation(self, equipment_client):
        """Test creating EquipmentClient."""
        assert equipment_client.equipment_type == EquipmentType.CNC_MILL
        assert equipment_client.equipment_id == "cnc1"
        assert equipment_client.connection_config["type"] == "network"
        assert equipment_client.is_connected is False
        assert equipment_client.is_monitoring is False

    @pytest.mark.unit
    def test_equipment_client_creation_string_type(self):
        """Test creating EquipmentClient with string equipment type."""
        client = EquipmentClient(
            equipment_type="3d_printer",
            equipment_id="printer1",
            connection_config={"type": "network"},
        )
        assert client.equipment_type == EquipmentType.THREE_D_PRINTER

    @pytest.mark.unit
    def test_connect_success(self, equipment_client):
        """Test connecting to equipment successfully."""
        result = equipment_client.connect()

        assert result is True
        assert equipment_client.is_connected is True

    @pytest.mark.unit
    def test_connect_already_connected(self, equipment_client):
        """Test connecting when already connected."""
        equipment_client.connect()
        result = equipment_client.connect()

        assert result is True  # Should return True even if already connected
        assert equipment_client.is_connected is True

    @pytest.mark.unit
    def test_disconnect_success(self, equipment_client):
        """Test disconnecting from equipment."""
        equipment_client.connect()
        assert equipment_client.is_connected is True

        equipment_client.disconnect()

        assert equipment_client.is_connected is False

    @pytest.mark.unit
    def test_disconnect_not_connected(self, equipment_client):
        """Test disconnecting when not connected."""
        # Should not raise error
        equipment_client.disconnect()
        assert equipment_client.is_connected is False

    @pytest.mark.unit
    def test_read_sensor_data_success(self, equipment_client):
        """Test reading sensor data successfully."""
        equipment_client.connect()

        data = equipment_client.read_sensor_data("sensor1")

        assert isinstance(data, dict)
        assert "sensor_id" in data
        assert data["sensor_id"] == "sensor1"
        assert "value" in data
        assert "timestamp" in data

    @pytest.mark.unit
    def test_read_sensor_data_not_connected(self, equipment_client):
        """Test reading sensor data when not connected."""
        with pytest.raises(ConnectionError, match="not connected"):
            equipment_client.read_sensor_data("sensor1")

    @pytest.mark.unit
    def test_send_command_success(self, equipment_client):
        """Test sending command successfully."""
        equipment_client.connect()

        result = equipment_client.send_command("start", {"speed": 100})

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "message" in result
        assert "timestamp" in result

    @pytest.mark.unit
    def test_send_command_not_connected(self, equipment_client):
        """Test sending command when not connected."""
        with pytest.raises(ConnectionError, match="not connected"):
            equipment_client.send_command("start")

    @pytest.mark.unit
    def test_get_equipment_status_success(self, equipment_client):
        """Test getting equipment status successfully."""
        equipment_client.connect()

        status = equipment_client.get_equipment_status()

        assert isinstance(status, EquipmentStatus)
        assert status.equipment_id == "cnc1"
        assert status.status in [s.value for s in EquipmentStatusValue]

    @pytest.mark.unit
    def test_get_equipment_status_not_connected(self, equipment_client):
        """Test getting equipment status when not connected."""
        with pytest.raises(ConnectionError, match="not connected"):
            equipment_client.get_equipment_status()

    @pytest.mark.unit
    def test_register_status_callback(self, equipment_client):
        """Test registering status callback."""
        callback_called = []

        def callback(status):
            callback_called.append(status)

        equipment_client.register_status_callback(callback)

        # Start monitoring to trigger callback
        equipment_client.connect()
        equipment_client.start_monitoring(update_interval=0.1)
        time.sleep(0.15)
        equipment_client.stop_monitoring()

        # Callback should have been called
        assert len(callback_called) >= 1
        assert isinstance(callback_called[0], EquipmentStatus)

    @pytest.mark.unit
    def test_unregister_status_callback(self, equipment_client):
        """Test unregistering status callback."""
        callback_called = []

        def callback(status):
            callback_called.append(status)

        equipment_client.register_status_callback(callback)
        equipment_client.unregister_status_callback(callback)

        # Start monitoring - callback should not be called
        equipment_client.connect()
        equipment_client.start_monitoring(update_interval=0.1)
        time.sleep(0.15)
        equipment_client.stop_monitoring()

        # Callback should not have been called
        assert len(callback_called) == 0

    @pytest.mark.unit
    def test_start_monitoring_success(self, equipment_client):
        """Test starting monitoring."""
        equipment_client.connect()

        equipment_client.start_monitoring(update_interval=0.1)

        assert equipment_client.is_monitoring is True

        # Clean up
        equipment_client.stop_monitoring()

    @pytest.mark.unit
    def test_start_monitoring_not_connected(self, equipment_client):
        """Test starting monitoring when not connected."""
        with pytest.raises(ConnectionError, match="not connected"):
            equipment_client.start_monitoring()

    @pytest.mark.unit
    def test_start_monitoring_already_active(self, equipment_client):
        """Test starting monitoring when already active."""
        equipment_client.connect()
        equipment_client.start_monitoring()

        # Should not raise error, just log warning
        equipment_client.start_monitoring()

        assert equipment_client.is_monitoring is True

        # Clean up
        equipment_client.stop_monitoring()

    @pytest.mark.unit
    def test_stop_monitoring(self, equipment_client):
        """Test stopping monitoring."""
        equipment_client.connect()
        equipment_client.start_monitoring()

        equipment_client.stop_monitoring()

        assert equipment_client.is_monitoring is False

    @pytest.mark.unit
    def test_stop_monitoring_not_active(self, equipment_client):
        """Test stopping monitoring when not active."""
        # Should not raise error
        equipment_client.stop_monitoring()
        assert equipment_client.is_monitoring is False

    @pytest.mark.unit
    def test_context_manager(self, equipment_client):
        """Test using EquipmentClient as context manager."""
        with equipment_client:
            assert equipment_client.is_connected is True

        assert equipment_client.is_connected is False

    @pytest.mark.unit
    def test_disconnect_stops_monitoring(self, equipment_client):
        """Test that disconnect stops monitoring."""
        equipment_client.connect()
        equipment_client.start_monitoring()

        equipment_client.disconnect()

        assert equipment_client.is_connected is False
        assert equipment_client.is_monitoring is False

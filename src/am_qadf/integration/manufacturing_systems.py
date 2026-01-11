"""
Manufacturing systems integration.

Provides client for integrating with manufacturing equipment (CNC machines, 3D printers, sensors, PLCs, etc.).
"""

import logging
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class EquipmentType(str, Enum):
    """Manufacturing equipment types."""

    CNC_MILL = "cnc_mill"
    CNC_LATHE = "cnc_lathe"
    THREE_D_PRINTER = "3d_printer"
    LASER_CUTTER = "laser_cutter"
    SENSOR = "sensor"
    PLC = "plc"
    ROBOT_ARM = "robot_arm"
    CMM = "cmm"  # Coordinate Measuring Machine
    OTHER = "other"


class EquipmentStatusValue(str, Enum):
    """Equipment status values."""

    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    PAUSED = "paused"


@dataclass
class EquipmentStatus:
    """Equipment status information."""

    equipment_id: str
    status: str  # EquipmentStatusValue
    current_operation: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EquipmentStatus":
        """Create from dictionary."""
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class EquipmentClient:
    """
    Client for manufacturing equipment integration.

    Provides methods for:
    - Connecting to and disconnecting from equipment
    - Reading sensor data
    - Sending commands to equipment
    - Monitoring equipment status
    - Handling equipment events
    """

    def __init__(
        self,
        equipment_type: str,
        equipment_id: str,
        connection_config: Dict[str, Any],
    ):
        """
        Initialize equipment client.

        Args:
            equipment_type: Type of equipment (EquipmentType enum value or string)
            equipment_id: Unique equipment identifier
            connection_config: Connection configuration dictionary
                Should contain connection-specific parameters:
                - For network devices: 'host', 'port', 'protocol' (http, tcp, modbus, etc.)
                - For serial devices: 'port', 'baudrate', 'timeout'
                - For file-based: 'file_path', 'format'
                - Authentication: 'username', 'password', 'api_key' (if needed)
        """
        self.equipment_type = EquipmentType(equipment_type) if isinstance(equipment_type, str) else equipment_type
        self.equipment_id = equipment_id
        self.connection_config = connection_config

        self._connected = False
        self._connection_lock = threading.Lock()
        self._status_callbacks: List[Callable[[EquipmentStatus], None]] = []
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None

        # Equipment-specific connection object (to be set by subclasses)
        self._connection = None

        logger.info(f"EquipmentClient initialized: {equipment_type} - {equipment_id}")

    def connect(self) -> bool:
        """
        Connect to manufacturing equipment.

        Returns:
            True if connection successful, False otherwise
        """
        with self._connection_lock:
            if self._connected:
                logger.warning(f"Equipment {self.equipment_id} already connected")
                return True

            try:
                # Implementation depends on equipment type and connection config
                # This is a base implementation that can be extended
                connection_type = self.connection_config.get("type", "network")

                if connection_type == "network":
                    self._connect_network()
                elif connection_type == "serial":
                    self._connect_serial()
                elif connection_type == "file":
                    self._connect_file()
                else:
                    logger.warning(f"Unknown connection type: {connection_type}, using mock connection")
                    self._connected = True  # Mock connection for testing

                if self._connected:
                    logger.info(f"Connected to equipment {self.equipment_id}")
                else:
                    logger.error(f"Failed to connect to equipment {self.equipment_id}")

                return self._connected

            except Exception as e:
                logger.error(f"Error connecting to equipment {self.equipment_id}: {e}")
                self._connected = False
                return False

    def _connect_network(self):
        """Connect via network (HTTP, TCP, Modbus, etc.)."""
        # Base implementation - should be extended by specific equipment types
        # For now, mark as connected (mock connection)
        self._connected = True
        logger.debug(f"Network connection to {self.equipment_id} established (mock)")

    def _connect_serial(self):
        """Connect via serial port."""
        # Base implementation - should be extended by specific equipment types
        # For now, mark as connected (mock connection)
        self._connected = True
        logger.debug(f"Serial connection to {self.equipment_id} established (mock)")

    def _connect_file(self):
        """Connect via file-based interface."""
        # Base implementation - should be extended by specific equipment types
        # For now, mark as connected (mock connection)
        self._connected = True
        logger.debug(f"File-based connection to {self.equipment_id} established (mock)")

    def disconnect(self) -> None:
        """Disconnect from manufacturing equipment."""
        with self._connection_lock:
            if not self._connected:
                logger.warning(f"Equipment {self.equipment_id} not connected")
                return

            # Stop monitoring
            self.stop_monitoring()

            # Close connection
            try:
                if self._connection:
                    if hasattr(self._connection, "close"):
                        self._connection.close()
                    self._connection = None

                self._connected = False
                logger.info(f"Disconnected from equipment {self.equipment_id}")
            except Exception as e:
                logger.error(f"Error disconnecting from equipment {self.equipment_id}: {e}")

    def read_sensor_data(self, sensor_id: str) -> Dict[str, Any]:
        """
        Read data from sensor.

        Args:
            sensor_id: Sensor identifier

        Returns:
            Dictionary containing sensor data:
                - value: Sensor reading value
                - unit: Unit of measurement
                - timestamp: Reading timestamp
                - status: Sensor status (ok, error, etc.)
                - metadata: Additional sensor-specific data

        Raises:
            ConnectionError: If not connected
            ValueError: If sensor not found
        """
        if not self._connected:
            raise ConnectionError(f"Equipment {self.equipment_id} not connected")

        try:
            # Base implementation - should be extended by specific equipment types
            # This is a mock implementation
            return {
                "sensor_id": sensor_id,
                "value": 0.0,
                "unit": "unknown",
                "timestamp": datetime.now().isoformat(),
                "status": "ok",
                "metadata": {},
            }
        except Exception as e:
            logger.error(f"Error reading sensor {sensor_id}: {e}")
            raise

    def send_command(
        self,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send command to equipment.

        Args:
            command: Command name (equipment-specific)
            parameters: Optional command parameters

        Returns:
            Dictionary containing command response:
                - success: Whether command was successful
                - message: Response message
                - data: Response data (if any)
                - timestamp: Command execution timestamp

        Raises:
            ConnectionError: If not connected
            ValueError: If command is invalid
        """
        if not self._connected:
            raise ConnectionError(f"Equipment {self.equipment_id} not connected")

        try:
            # Base implementation - should be extended by specific equipment types
            # This is a mock implementation
            logger.info(f"Sending command '{command}' to equipment {self.equipment_id}")

            return {
                "success": True,
                "message": f"Command '{command}' executed successfully",
                "data": parameters or {},
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error sending command '{command}': {e}")
            return {
                "success": False,
                "message": str(e),
                "data": {},
                "timestamp": datetime.now().isoformat(),
            }

    def get_equipment_status(self) -> EquipmentStatus:
        """
        Get current equipment status.

        Returns:
            EquipmentStatus object

        Raises:
            ConnectionError: If not connected
        """
        if not self._connected:
            raise ConnectionError(f"Equipment {self.equipment_id} not connected")

        try:
            # Base implementation - should be extended by specific equipment types
            # This is a mock implementation
            return EquipmentStatus(
                equipment_id=self.equipment_id,
                status=EquipmentStatusValue.IDLE.value,
                current_operation=None,
                progress=0.0,
                error_message=None,
                timestamp=datetime.now(),
                metadata={},
            )
        except Exception as e:
            logger.error(f"Error getting equipment status: {e}")
            return EquipmentStatus(
                equipment_id=self.equipment_id,
                status=EquipmentStatusValue.ERROR.value,
                error_message=str(e),
                timestamp=datetime.now(),
            )

    def register_status_callback(self, callback: Callable[[EquipmentStatus], None]) -> None:
        """
        Register callback for status updates.

        Args:
            callback: Function to call with EquipmentStatus object
        """
        self._status_callbacks.append(callback)
        logger.debug(f"Registered status callback for equipment {self.equipment_id}")

    def unregister_status_callback(self, callback: Callable[[EquipmentStatus], None]) -> None:
        """Unregister status callback."""
        if callback in self._status_callbacks:
            self._status_callbacks.remove(callback)
            logger.debug(f"Unregistered status callback for equipment {self.equipment_id}")

    def start_monitoring(self, update_interval: float = 5.0) -> None:
        """
        Start continuous status monitoring.

        Args:
            update_interval: Status update interval in seconds
        """
        if self._monitoring_active:
            logger.warning(f"Monitoring already active for equipment {self.equipment_id}")
            return

        if not self._connected:
            raise ConnectionError(f"Equipment {self.equipment_id} not connected")

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(update_interval,),
            daemon=True,
        )
        self._monitoring_thread.start()
        logger.info(f"Started monitoring for equipment {self.equipment_id}")

    def stop_monitoring(self) -> None:
        """Stop continuous status monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
            self._monitoring_thread = None
        logger.info(f"Stopped monitoring for equipment {self.equipment_id}")

    def _monitoring_loop(self, update_interval: float) -> None:
        """Internal monitoring loop."""
        while self._monitoring_active:
            try:
                status = self.get_equipment_status()

                # Notify callbacks
                for callback in self._status_callbacks:
                    try:
                        callback(status)
                    except Exception as e:
                        logger.error(f"Error in status callback: {e}")

                time.sleep(update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(update_interval)

    @property
    def is_connected(self) -> bool:
        """Check if equipment is connected."""
        return self._connected

    @property
    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._monitoring_active

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

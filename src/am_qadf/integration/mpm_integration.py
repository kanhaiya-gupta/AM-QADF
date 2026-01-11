"""
MPM (Manufacturing Process Management) system integration.

Provides client for integrating with MPM systems via REST API.
"""

import logging
import requests
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class MPMStatus(str, Enum):
    """MPM process status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class MPMProcessData:
    """Process data from MPM system."""

    process_id: str
    build_id: str
    material: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = MPMStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.start_time:
            data["start_time"] = self.start_time.isoformat()
        if self.end_time:
            data["end_time"] = self.end_time.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MPMProcessData":
        """Create from dictionary."""
        if "start_time" in data and isinstance(data["start_time"], str):
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if "end_time" in data and isinstance(data["end_time"], str):
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        return cls(**data)


class MPMClient:
    """
    Client for MPM system integration.

    Provides methods for:
    - Retrieving process data from MPM system
    - Updating process status
    - Submitting quality assessment results
    - Getting optimized process parameters
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        retry_on_failure: bool = True,
        max_retries: int = 3,
    ):
        """
        Initialize MPM client.

        Args:
            base_url: Base URL of MPM system API
            api_key: API key for authentication (optional)
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            retry_on_failure: Whether to retry on failure
            max_retries: Maximum number of retries
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries

        # Session for connection pooling
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )

        logger.info(f"MPMClient initialized with base_url: {base_url}")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to MPM API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (relative to base_url)
            data: Request body data
            params: Query parameters

        Returns:
            Response data as dictionary

        Raises:
            requests.RequestException: If request fails
            ValueError: If response is invalid
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        attempts = 0
        last_exception = None

        while attempts < (self.max_retries + 1 if self.retry_on_failure else 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                response.raise_for_status()
                return response.json() if response.content else {}

            except requests.exceptions.RequestException as e:
                last_exception = e
                attempts += 1
                if not self.retry_on_failure or attempts > self.max_retries:
                    logger.error(f"MPM API request failed after {attempts} attempts: {e}")
                    raise
                logger.warning(f"MPM API request failed, retrying ({attempts}/{self.max_retries}): {e}")
                continue

        raise last_exception

    def get_process_data(self, process_id: str) -> MPMProcessData:
        """
        Get process data from MPM system.

        Args:
            process_id: Process identifier

        Returns:
            MPMProcessData object

        Raises:
            requests.RequestException: If request fails
            ValueError: If process not found or invalid response
        """
        try:
            response_data = self._make_request("GET", f"/api/v1/processes/{process_id}")
            return MPMProcessData.from_dict(response_data)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Process {process_id} not found") from e
            raise

    def update_process_status(
        self,
        process_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update process status in MPM system.

        Args:
            process_id: Process identifier
            status: New status (should be valid MPMStatus value)
            metadata: Optional metadata to include

        Returns:
            True if successful

        Raises:
            requests.RequestException: If request fails
            ValueError: If status is invalid
        """
        if status not in [s.value for s in MPMStatus]:
            raise ValueError(f"Invalid status: {status}. Must be one of {[s.value for s in MPMStatus]}")

        data = {"status": status}
        if metadata:
            data["metadata"] = metadata

        try:
            self._make_request("PUT", f"/api/v1/processes/{process_id}/status", data=data)
            logger.info(f"Updated process {process_id} status to {status}")
            return True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Process {process_id} not found") from e
            raise

    def submit_quality_results(
        self,
        process_id: str,
        results: Dict[str, Any],
    ) -> bool:
        """
        Submit quality assessment results to MPM system.

        Args:
            process_id: Process identifier
            results: Quality assessment results dictionary
                Should contain keys like: overall_score, quality_scores, defects, etc.

        Returns:
            True if successful

        Raises:
            requests.RequestException: If request fails
            ValueError: If process not found
        """
        data = {
            "process_id": process_id,
            "results": results,
            "submitted_at": datetime.now().isoformat(),
        }

        try:
            self._make_request("POST", f"/api/v1/processes/{process_id}/quality", data=data)
            logger.info(f"Submitted quality results for process {process_id}")
            return True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Process {process_id} not found") from e
            raise

    def get_process_parameters(self, process_id: str) -> Dict[str, Any]:
        """
        Get optimized process parameters from MPM system.

        Args:
            process_id: Process identifier

        Returns:
            Dictionary containing process parameters
                Keys may include: laser_power, scan_speed, layer_thickness, etc.

        Raises:
            requests.RequestException: If request fails
            ValueError: If process not found
        """
        try:
            response_data = self._make_request("GET", f"/api/v1/processes/{process_id}/parameters")
            return response_data.get("parameters", {})
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Process {process_id} not found") from e
            raise

    def list_processes(
        self,
        build_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MPMProcessData]:
        """
        List processes from MPM system.

        Args:
            build_id: Optional filter by build ID
            status: Optional filter by status
            limit: Maximum number of processes to return
            offset: Offset for pagination

        Returns:
            List of MPMProcessData objects
        """
        params = {
            "limit": limit,
            "offset": offset,
        }
        if build_id:
            params["build_id"] = build_id
        if status:
            params["status"] = status

        try:
            response_data = self._make_request("GET", "/api/v1/processes", params=params)
            processes = response_data.get("processes", [])
            return [MPMProcessData.from_dict(p) for p in processes]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list processes: {e}")
            return []

    def health_check(self) -> bool:
        """
        Check MPM system health.

        Returns:
            True if system is healthy
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5.0,
                verify=self.verify_ssl,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def close(self):
        """Close client session."""
        self.session.close()
        logger.info("MPMClient session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

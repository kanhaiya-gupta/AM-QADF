"""
API Gateway for industrial access to AM-QADF functionality.

Provides REST API gateway with endpoint registration, middleware support,
and request/response handling.
"""

import logging
import json
import uuid
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Tuple
from enum import Enum
from abc import ABC, abstractmethod

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

logger = logging.getLogger(__name__)


class HTTPMethod(str, Enum):
    """HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


@dataclass
class APIRequest:
    """API request object."""

    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    path_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    client_ip: Optional[str] = None


@dataclass
class APIResponse:
    """API response object."""

    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "status_code": self.status_code,
            "headers": self.headers,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.body:
            data["body"] = self.body
        if self.error:
            data["error"] = self.error
        return data


class APIEndpoint:
    """API endpoint definition."""

    def __init__(
        self,
        path: str,
        method: str,
        handler: Callable[[APIRequest], APIResponse],
        summary: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        requires_auth: bool = True,
        rate_limit: Optional[int] = None,
    ):
        """
        Initialize API endpoint.

        Args:
            path: Endpoint path (e.g., '/api/v1/health')
            method: HTTP method (GET, POST, etc.)
            handler: Handler function that takes APIRequest and returns APIResponse
            summary: Short summary of endpoint
            description: Detailed description
            tags: Endpoint tags for grouping
            requires_auth: Whether authentication is required
            rate_limit: Optional rate limit (requests per minute)
        """
        self.path = path
        self.method = HTTPMethod(method.upper())
        self.handler = handler
        self.summary = summary
        self.description = description
        self.tags = tags or []
        self.requires_auth = requires_auth
        self.rate_limit = rate_limit

    def matches(self, method: str, path: str) -> Tuple[bool, Dict[str, str]]:
        """
        Check if endpoint matches request.

        Args:
            method: HTTP method
            path: Request path

        Returns:
            Tuple of (matches, path_params)
        """
        if HTTPMethod(method.upper()) != self.method:
            return False, {}

        # Simple path matching (supports :param style)
        endpoint_parts = self.path.rstrip("/").split("/")
        request_parts = path.rstrip("/").split("/")

        if len(endpoint_parts) != len(request_parts):
            return False, {}

        path_params = {}
        for ep_part, req_part in zip(endpoint_parts, request_parts):
            if ep_part.startswith(":"):
                # Path parameter
                param_name = ep_part[1:]
                path_params[param_name] = req_part
            elif ep_part != req_part:
                # Path segment doesn't match
                return False, {}

        return True, path_params


class APIMiddleware(ABC):
    """Base class for API middleware."""

    @abstractmethod
    def process_request(self, request: APIRequest) -> Optional[APIResponse]:
        """
        Process request before handler.

        Args:
            request: API request

        Returns:
            Optional APIResponse if request should be rejected, None to continue
        """
        pass

    @abstractmethod
    def process_response(self, request: APIRequest, response: APIResponse) -> APIResponse:
        """
        Process response after handler.

        Args:
            request: API request
            response: API response

        Returns:
            Modified API response
        """
        pass


class APIGateway:
    """
    REST API gateway for industrial access.

    Provides:
    - Endpoint registration and routing
    - Middleware support (authentication, logging, rate limiting)
    - Request/response handling
    - API versioning
    - Health check and metrics endpoints
    """

    def __init__(
        self,
        base_path: str = "/api/v1",
        enable_cors: bool = True,
        enable_logging: bool = True,
        default_timeout: float = 30.0,
    ):
        """
        Initialize API gateway.

        Args:
            base_path: Base path for API (e.g., '/api/v1')
            enable_cors: Enable CORS headers
            enable_logging: Enable request/response logging
            default_timeout: Default request timeout in seconds
        """
        self.base_path = base_path.rstrip("/")
        self.enable_cors = enable_cors
        self.enable_logging = enable_logging
        self.default_timeout = default_timeout

        # Endpoints and middleware
        self.endpoints: List[APIEndpoint] = []
        self.middleware: List[APIMiddleware] = []

        # Request tracking
        self.request_count = 0
        self.error_count = 0
        self._lock = threading.Lock()

        logger.info(f"APIGateway initialized with base_path: {base_path}")

    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        """
        Register API endpoint.

        Args:
            endpoint: APIEndpoint instance
        """
        self.endpoints.append(endpoint)
        logger.info(f"Registered endpoint: {endpoint.method.value} {endpoint.path}")

    def add_middleware(self, middleware: APIMiddleware) -> None:
        """
        Add middleware to API gateway.

        Args:
            middleware: APIMiddleware instance
        """
        self.middleware.append(middleware)
        logger.info(f"Added middleware: {middleware.__class__.__name__}")

    def find_endpoint(self, method: str, path: str) -> Tuple[Optional[APIEndpoint], Dict[str, str]]:
        """
        Find endpoint matching request.

        Args:
            method: HTTP method
            path: Request path (relative to base_path)

        Returns:
            Tuple of (endpoint, path_params) or (None, {}) if not found
        """
        # Remove base_path if present
        if path.startswith(self.base_path):
            path = path[len(self.base_path) :]

        path = path or "/"

        # Find matching endpoint
        for endpoint in self.endpoints:
            matches, path_params = endpoint.matches(method, path)
            if matches:
                return endpoint, path_params

        return None, {}

    def handle_request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> APIResponse:
        """
        Handle API request.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers
            query_params: Query parameters
            body: Request body
            user_id: User ID (if authenticated)
            client_ip: Client IP address

        Returns:
            APIResponse object
        """
        with self._lock:
            self.request_count += 1

        # Create request object
        request = APIRequest(
            method=method,
            path=path,
            headers=headers or {},
            query_params=query_params or {},
            body=body,
            user_id=user_id,
            client_ip=client_ip,
        )

        if self.enable_logging:
            logger.info(f"API Request: {method} {path} (ID: {request.request_id})")

        # Process request through middleware
        for middleware in self.middleware:
            try:
                response = middleware.process_request(request)
                if response:
                    # Middleware rejected request
                    response.request_id = request.request_id
                    if self.enable_logging:
                        logger.warning(f"Request rejected by middleware: {response.status_code}")
                    return response
            except Exception as e:
                logger.error(f"Error in middleware {middleware.__class__.__name__}: {e}")
                with self._lock:
                    self.error_count += 1
                return self._create_error_response(
                    status_code=500,
                    message="Internal server error in middleware",
                    request_id=request.request_id,
                )

        # Find endpoint
        endpoint, path_params = self.find_endpoint(method, path)
        if not endpoint:
            if self.enable_logging:
                logger.warning(f"Endpoint not found: {method} {path}")
            with self._lock:
                self.error_count += 1
            return self._create_error_response(
                status_code=404,
                message=f"Endpoint not found: {method} {path}",
                request_id=request.request_id,
            )

        # Add path parameters to request
        request.path_params = path_params

        # Call handler
        try:
            response = endpoint.handler(request)
            response.request_id = request.request_id

            # Add CORS headers if enabled
            if self.enable_cors:
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"

            # Process response through middleware (in reverse order)
            for middleware in reversed(self.middleware):
                try:
                    response = middleware.process_response(request, response)
                except Exception as e:
                    logger.error(f"Error in middleware {middleware.__class__.__name__}: {e}")

            if self.enable_logging:
                logger.info(f"API Response: {response.status_code} (ID: {request.request_id})")

            if response.status_code >= 400:
                with self._lock:
                    self.error_count += 1

            return response

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            with self._lock:
                self.error_count += 1
            return self._create_error_response(
                status_code=500,
                message="Internal server error",
                details=str(e),
                request_id=request.request_id,
            )

    def _create_error_response(
        self,
        status_code: int,
        message: str,
        details: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> APIResponse:
        """Create error response."""
        error = {
            "code": f"AM_QADF_{status_code:03d}",
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        if details:
            error["details"] = details

        return APIResponse(
            status_code=status_code,
            error=error,
            request_id=request_id,
        )

    def get_health_status(self) -> Dict[str, Any]:
        """Get API gateway health status."""
        return {
            "status": "healthy",
            "uptime": "N/A",  # Would track actual uptime
            "request_count": self.request_count,
            "error_count": self.error_count,
            "endpoint_count": len(self.endpoints),
            "middleware_count": len(self.middleware),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get API gateway metrics."""
        return {
            "requests_total": self.request_count,
            "errors_total": self.error_count,
            "endpoints_total": len(self.endpoints),
            "middleware_total": len(self.middleware),
        }

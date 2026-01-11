"""
Unit tests for API Gateway.

Tests for APIGateway, APIEndpoint, and APIMiddleware.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from am_qadf.integration.api_gateway import (
    APIGateway,
    APIEndpoint,
    APIMiddleware,
    APIRequest,
    APIResponse,
    HTTPMethod,
)


class TestAPIRequest:
    """Test suite for APIRequest dataclass."""

    @pytest.mark.unit
    def test_api_request_creation(self):
        """Test creating APIRequest."""
        request = APIRequest(
            method="GET",
            path="/api/v1/health",
            headers={"Authorization": "Bearer token123"},
            query_params={"limit": 10},
            body={"data": "test"},
        )

        assert request.method == "GET"
        assert request.path == "/api/v1/health"
        assert request.headers["Authorization"] == "Bearer token123"
        assert request.query_params["limit"] == 10
        assert request.body["data"] == "test"
        assert request.request_id is not None
        assert isinstance(request.timestamp, datetime)


class TestAPIResponse:
    """Test suite for APIResponse dataclass."""

    @pytest.mark.unit
    def test_api_response_creation(self):
        """Test creating APIResponse."""
        response = APIResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={"result": "success"},
        )

        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        assert response.body["result"] == "success"
        assert response.error is None

    @pytest.mark.unit
    def test_api_response_with_error(self):
        """Test creating APIResponse with error."""
        response = APIResponse(
            status_code=404,
            error={"code": "NOT_FOUND", "message": "Resource not found"},
        )

        assert response.status_code == 404
        assert response.error["code"] == "NOT_FOUND"
        assert response.body is None

    @pytest.mark.unit
    def test_api_response_to_dict(self):
        """Test converting APIResponse to dictionary."""
        response = APIResponse(
            status_code=200,
            body={"data": "test"},
            request_id="req123",
        )

        response_dict = response.to_dict()

        assert isinstance(response_dict, dict)
        assert response_dict["status_code"] == 200
        assert response_dict["request_id"] == "req123"
        assert isinstance(response_dict["timestamp"], str)


class TestAPIEndpoint:
    """Test suite for APIEndpoint class."""

    @pytest.mark.unit
    def test_api_endpoint_creation(self):
        """Test creating APIEndpoint."""

        def handler(request):
            return APIResponse(status_code=200, body={"result": "ok"})

        endpoint = APIEndpoint(
            path="/api/v1/health",
            method="GET",
            handler=handler,
            summary="Health check",
            description="Check API health status",
            tags=["health"],
            requires_auth=False,
            rate_limit=100,
        )

        assert endpoint.path == "/api/v1/health"
        assert endpoint.method == HTTPMethod.GET
        assert endpoint.handler == handler
        assert endpoint.summary == "Health check"
        assert endpoint.requires_auth is False
        assert endpoint.rate_limit == 100

    @pytest.mark.unit
    def test_api_endpoint_matches_exact_path(self):
        """Test endpoint matching with exact path."""

        def handler(request):
            return APIResponse(status_code=200)

        endpoint = APIEndpoint(path="/api/v1/health", method="GET", handler=handler)

        matches, params = endpoint.matches("GET", "/api/v1/health")
        assert matches is True
        assert params == {}

    @pytest.mark.unit
    def test_api_endpoint_matches_wrong_method(self):
        """Test endpoint matching with wrong HTTP method."""

        def handler(request):
            return APIResponse(status_code=200)

        endpoint = APIEndpoint(path="/api/v1/health", method="GET", handler=handler)

        matches, params = endpoint.matches("POST", "/api/v1/health")
        assert matches is False

    @pytest.mark.unit
    def test_api_endpoint_matches_path_parameters(self):
        """Test endpoint matching with path parameters."""

        def handler(request):
            return APIResponse(status_code=200)

        endpoint = APIEndpoint(path="/api/v1/processes/:process_id", method="GET", handler=handler)

        matches, params = endpoint.matches("GET", "/api/v1/processes/proc123")
        assert matches is True
        assert params["process_id"] == "proc123"

    @pytest.mark.unit
    def test_api_endpoint_matches_multiple_path_parameters(self):
        """Test endpoint matching with multiple path parameters."""

        def handler(request):
            return APIResponse(status_code=200)

        endpoint = APIEndpoint(path="/api/v1/builds/:build_id/processes/:process_id", method="GET", handler=handler)

        matches, params = endpoint.matches("GET", "/api/v1/builds/build123/processes/proc456")
        assert matches is True
        assert params["build_id"] == "build123"
        assert params["process_id"] == "proc456"


class TestAPIMiddleware:
    """Test suite for APIMiddleware base class."""

    @pytest.mark.unit
    def test_api_middleware_abstract(self):
        """Test that APIMiddleware is abstract."""
        with pytest.raises(TypeError):
            APIMiddleware()


class MockMiddleware(APIMiddleware):
    """Mock middleware for testing."""

    def __init__(self):
        self.request_processed = False
        self.response_processed = False

    def process_request(self, request):
        """Process request."""
        self.request_processed = True
        return None

    def process_response(self, request, response):
        """Process response."""
        self.response_processed = True
        return response


class TestAPIGateway:
    """Test suite for APIGateway class."""

    @pytest.fixture
    def api_gateway(self):
        """Create an APIGateway instance."""
        return APIGateway(base_path="/api/v1")

    @pytest.mark.unit
    def test_api_gateway_creation(self, api_gateway):
        """Test creating APIGateway."""
        assert api_gateway.base_path == "/api/v1"
        assert api_gateway.enable_cors is True
        assert api_gateway.enable_logging is True
        assert len(api_gateway.endpoints) == 0
        assert len(api_gateway.middleware) == 0
        assert api_gateway.request_count == 0

    @pytest.mark.unit
    def test_api_gateway_creation_custom_settings(self):
        """Test creating APIGateway with custom settings."""
        gateway = APIGateway(
            base_path="/api/v2",
            enable_cors=False,
            enable_logging=False,
            default_timeout=60.0,
        )

        assert gateway.base_path == "/api/v2"
        assert gateway.enable_cors is False
        assert gateway.enable_logging is False
        assert gateway.default_timeout == 60.0

    @pytest.mark.unit
    def test_register_endpoint(self, api_gateway):
        """Test registering endpoint."""

        def handler(request):
            return APIResponse(status_code=200)

        endpoint = APIEndpoint(path="/health", method="GET", handler=handler)
        api_gateway.register_endpoint(endpoint)

        assert len(api_gateway.endpoints) == 1
        assert api_gateway.endpoints[0] == endpoint

    @pytest.mark.unit
    def test_add_middleware(self, api_gateway):
        """Test adding middleware."""
        middleware = MockMiddleware()
        api_gateway.add_middleware(middleware)

        assert len(api_gateway.middleware) == 1
        assert api_gateway.middleware[0] == middleware

    @pytest.mark.unit
    def test_find_endpoint_success(self, api_gateway):
        """Test finding endpoint."""

        def handler(request):
            return APIResponse(status_code=200)

        endpoint = APIEndpoint(path="/health", method="GET", handler=handler)
        api_gateway.register_endpoint(endpoint)

        found_endpoint, params = api_gateway.find_endpoint("GET", "/api/v1/health")

        assert found_endpoint == endpoint
        assert params == {}

    @pytest.mark.unit
    def test_find_endpoint_with_path_params(self, api_gateway):
        """Test finding endpoint with path parameters."""

        def handler(request):
            return APIResponse(status_code=200)

        endpoint = APIEndpoint(path="/processes/:process_id", method="GET", handler=handler)
        api_gateway.register_endpoint(endpoint)

        found_endpoint, params = api_gateway.find_endpoint("GET", "/api/v1/processes/proc123")

        assert found_endpoint == endpoint
        assert params["process_id"] == "proc123"

    @pytest.mark.unit
    def test_find_endpoint_not_found(self, api_gateway):
        """Test finding endpoint that doesn't exist."""
        found_endpoint, params = api_gateway.find_endpoint("GET", "/api/v1/nonexistent")

        assert found_endpoint is None
        assert params == {}

    @pytest.mark.unit
    def test_handle_request_success(self, api_gateway):
        """Test handling request successfully."""

        def handler(request):
            return APIResponse(status_code=200, body={"result": "success"})

        endpoint = APIEndpoint(path="/health", method="GET", handler=handler)
        api_gateway.register_endpoint(endpoint)

        response = api_gateway.handle_request("GET", "/api/v1/health")

        assert response.status_code == 200
        assert response.body["result"] == "success"
        assert api_gateway.request_count == 1

    @pytest.mark.unit
    def test_handle_request_with_middleware(self, api_gateway):
        """Test handling request with middleware."""

        def handler(request):
            return APIResponse(status_code=200)

        endpoint = APIEndpoint(path="/health", method="GET", handler=handler)
        api_gateway.register_endpoint(endpoint)

        middleware = MockMiddleware()
        api_gateway.add_middleware(middleware)

        response = api_gateway.handle_request("GET", "/api/v1/health")

        assert response.status_code == 200
        assert middleware.request_processed is True
        assert middleware.response_processed is True

    @pytest.mark.unit
    def test_handle_request_endpoint_not_found(self, api_gateway):
        """Test handling request for non-existent endpoint."""
        response = api_gateway.handle_request("GET", "/api/v1/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.error["message"].lower()
        assert api_gateway.error_count == 1

    @pytest.mark.unit
    def test_handle_request_handler_exception(self, api_gateway):
        """Test handling request when handler raises exception."""

        def handler(request):
            raise ValueError("Handler error")

        endpoint = APIEndpoint(path="/health", method="GET", handler=handler)
        api_gateway.register_endpoint(endpoint)

        response = api_gateway.handle_request("GET", "/api/v1/health")

        assert response.status_code == 500
        assert response.error is not None
        assert api_gateway.error_count == 1

    @pytest.mark.unit
    def test_handle_request_cors_headers(self, api_gateway):
        """Test CORS headers are added."""

        def handler(request):
            return APIResponse(status_code=200)

        endpoint = APIEndpoint(path="/health", method="GET", handler=handler)
        api_gateway.register_endpoint(endpoint)

        response = api_gateway.handle_request("GET", "/api/v1/health")

        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == "*"

    @pytest.mark.unit
    def test_handle_request_no_cors(self):
        """Test CORS headers are not added when disabled."""
        gateway = APIGateway(enable_cors=False)

        def handler(request):
            return APIResponse(status_code=200)

        endpoint = APIEndpoint(path="/health", method="GET", handler=handler)
        gateway.register_endpoint(endpoint)

        response = gateway.handle_request("GET", "/api/v1/health")

        assert "Access-Control-Allow-Origin" not in response.headers

    @pytest.mark.unit
    def test_get_health_status(self, api_gateway):
        """Test getting health status."""
        health = api_gateway.get_health_status()

        assert isinstance(health, dict)
        assert health["status"] == "healthy"
        assert health["request_count"] == 0
        assert health["endpoint_count"] == 0

    @pytest.mark.unit
    def test_get_metrics(self, api_gateway):
        """Test getting metrics."""
        metrics = api_gateway.get_metrics()

        assert isinstance(metrics, dict)
        assert "requests_total" in metrics
        assert "errors_total" in metrics
        assert "endpoints_total" in metrics

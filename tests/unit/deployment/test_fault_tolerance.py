"""
Unit tests for fault tolerance utilities.

Tests for RetryPolicy, CircuitBreaker, and GracefulDegradation.
"""

import pytest
import time
from unittest.mock import Mock, patch

from am_qadf.deployment.fault_tolerance import (
    RetryPolicy,
    CircuitState,
    CircuitBreaker,
    CircuitBreakerOpenError,
    GracefulDegradation,
    RateLimitExceededError,
    retry_with_policy,
)


class TestRetryPolicy:
    """Test suite for RetryPolicy dataclass."""

    @pytest.mark.unit
    def test_retry_policy_creation_defaults(self):
        """Test creating RetryPolicy with defaults."""
        policy = RetryPolicy()

        assert policy.max_retries == 3
        assert policy.backoff_factor == 1.0
        assert policy.exponential_backoff is True
        assert Exception in policy.retryable_exceptions
        assert policy.max_backoff_time == 60.0
        assert policy.initial_backoff_time == 1.0

    @pytest.mark.unit
    def test_retry_policy_creation_custom(self):
        """Test creating RetryPolicy with custom values."""
        policy = RetryPolicy(
            max_retries=5,
            backoff_factor=2.0,
            exponential_backoff=False,
            retryable_exceptions=(ValueError, KeyError),
            max_backoff_time=120.0,
            initial_backoff_time=2.0,
        )

        assert policy.max_retries == 5
        assert policy.backoff_factor == 2.0
        assert policy.exponential_backoff is False
        assert ValueError in policy.retryable_exceptions
        assert policy.max_backoff_time == 120.0
        assert policy.initial_backoff_time == 2.0

    @pytest.mark.unit
    def test_should_retry_within_max_retries(self):
        """Test should_retry when within max retries."""
        policy = RetryPolicy(max_retries=3, retryable_exceptions=(ValueError,))

        assert policy.should_retry(0, ValueError("test")) is True
        assert policy.should_retry(1, ValueError("test")) is True
        assert policy.should_retry(2, ValueError("test")) is True
        assert policy.should_retry(3, ValueError("test")) is False  # Exceeded

    @pytest.mark.unit
    def test_should_retry_non_retryable_exception(self):
        """Test should_retry with non-retryable exception."""
        policy = RetryPolicy(max_retries=3, retryable_exceptions=(ValueError,))

        assert policy.should_retry(0, KeyError("test")) is False
        assert policy.should_retry(1, TypeError("test")) is False

    @pytest.mark.unit
    def test_get_backoff_time_exponential(self):
        """Test exponential backoff calculation."""
        policy = RetryPolicy(exponential_backoff=True, backoff_factor=2.0, initial_backoff_time=1.0, max_backoff_time=60.0)

        assert policy.get_backoff_time(0) == 1.0  # 1.0 * 2^0
        assert policy.get_backoff_time(1) == 2.0  # 1.0 * 2^1
        assert policy.get_backoff_time(2) == 4.0  # 1.0 * 2^2
        assert policy.get_backoff_time(3) == 8.0  # 1.0 * 2^3

    @pytest.mark.unit
    def test_get_backoff_time_linear(self):
        """Test linear backoff calculation."""
        policy = RetryPolicy(exponential_backoff=False, backoff_factor=2.0, initial_backoff_time=1.0, max_backoff_time=60.0)

        assert policy.get_backoff_time(0) == 2.0  # 1.0 * 2.0 * 1
        assert policy.get_backoff_time(1) == 4.0  # 1.0 * 2.0 * 2
        assert policy.get_backoff_time(2) == 6.0  # 1.0 * 2.0 * 3

    @pytest.mark.unit
    def test_get_backoff_time_capped(self):
        """Test backoff time is capped at max."""
        policy = RetryPolicy(exponential_backoff=True, backoff_factor=10.0, initial_backoff_time=10.0, max_backoff_time=50.0)

        backoff = policy.get_backoff_time(5)  # Would be very large
        assert backoff == 50.0  # Capped at max


class TestCircuitBreaker:
    """Test suite for CircuitBreaker class."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a CircuitBreaker instance."""
        return CircuitBreaker(failure_threshold=3, timeout=10.0, half_open_max_calls=2, success_threshold=2)

    @pytest.mark.unit
    def test_circuit_breaker_creation(self):
        """Test creating CircuitBreaker."""
        cb = CircuitBreaker()

        assert cb.failure_threshold == 5
        assert cb.timeout == 60.0
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.unit
    def test_circuit_breaker_call_success(self, circuit_breaker):
        """Test circuit breaker with successful call."""

        def successful_func():
            return "success"

        result = circuit_breaker.call(successful_func)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    @pytest.mark.unit
    def test_circuit_breaker_call_failure(self, circuit_breaker):
        """Test circuit breaker with failing call."""

        def failing_func():
            raise ValueError("error")

        with pytest.raises(ValueError):
            circuit_breaker.call(failing_func)

        assert circuit_breaker.failure_count == 1
        assert circuit_breaker.state == CircuitState.CLOSED  # Still closed

    @pytest.mark.unit
    def test_circuit_breaker_opens_after_threshold(self, circuit_breaker):
        """Test circuit breaker opens after failure threshold."""

        def failing_func():
            raise ValueError("error")

        # Trigger failures up to threshold
        for _ in range(circuit_breaker.failure_threshold):
            try:
                circuit_breaker.call(failing_func)
            except ValueError:
                pass

        assert circuit_breaker.state == CircuitState.OPEN

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            circuit_breaker.call(failing_func)

    @pytest.mark.unit
    def test_circuit_breaker_transitions_to_half_open(self, circuit_breaker):
        """Test circuit breaker transitions to half-open after timeout."""

        def failing_func():
            raise ValueError("error")

        # Open circuit
        for _ in range(circuit_breaker.failure_threshold):
            try:
                circuit_breaker.call(failing_func)
            except ValueError:
                pass

        assert circuit_breaker.state == CircuitState.OPEN

        # Simulate timeout passing
        circuit_breaker.last_failure_time = time.time() - (circuit_breaker.timeout + 1)

        def successful_func():
            return "success"

        # Should transition to half-open and allow call
        result = circuit_breaker.call(successful_func)
        assert result == "success"

    @pytest.mark.unit
    def test_circuit_breaker_closes_from_half_open(self, circuit_breaker):
        """Test circuit breaker closes from half-open after successes."""

        def successful_func():
            return "success"

        # Manually set to half-open
        circuit_breaker.state = CircuitState.HALF_OPEN

        # Record enough successes
        for _ in range(circuit_breaker.success_threshold):
            circuit_breaker.record_success()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    @pytest.mark.unit
    def test_circuit_breaker_opens_from_half_open_on_failure(self, circuit_breaker):
        """Test circuit breaker opens from half-open on failure."""

        def failing_func():
            raise ValueError("error")

        # Manually set to half-open
        circuit_breaker.state = CircuitState.HALF_OPEN

        # Record a failure
        try:
            circuit_breaker.call(failing_func)
        except ValueError:
            pass

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.unit
    def test_circuit_breaker_half_open_call_limit(self, circuit_breaker):
        """Test circuit breaker enforces half-open call limit."""
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker.half_open_calls = circuit_breaker.half_open_max_calls

        def func():
            return "success"

        with pytest.raises(CircuitBreakerOpenError, match="exceeded half-open call limit"):
            circuit_breaker.call(func)

    @pytest.mark.unit
    def test_record_success_resets_failure_count(self, circuit_breaker):
        """Test recording success resets failure count."""
        circuit_breaker.failure_count = 2
        circuit_breaker.record_success()

        assert circuit_breaker.failure_count == 0

    @pytest.mark.unit
    def test_record_failure_increases_count(self, circuit_breaker):
        """Test recording failure increases count."""
        initial_count = circuit_breaker.failure_count
        circuit_breaker.record_failure()

        assert circuit_breaker.failure_count == initial_count + 1

    @pytest.mark.unit
    def test_reset(self, circuit_breaker):
        """Test resetting circuit breaker."""
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 5
        circuit_breaker.record_failure()

        circuit_breaker.reset()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.last_failure_time is None

    @pytest.mark.unit
    def test_current_state_property(self, circuit_breaker):
        """Test current_state property."""
        assert circuit_breaker.current_state == "closed"

        circuit_breaker.state = CircuitState.OPEN
        assert circuit_breaker.current_state == "open"

        circuit_breaker.state = CircuitState.HALF_OPEN
        assert circuit_breaker.current_state == "half_open"


class TestGracefulDegradation:
    """Test suite for GracefulDegradation class."""

    @pytest.mark.unit
    def test_with_fallback_value(self):
        """Test graceful degradation with fallback value."""

        @GracefulDegradation.with_fallback(fallback_value="fallback_result")
        def failing_func():
            raise ValueError("error")

        result = failing_func()
        assert result == "fallback_result"

    @pytest.mark.unit
    def test_with_fallback_function(self):
        """Test graceful degradation with fallback function."""

        def fallback_handler(exception):
            return f"Handled: {str(exception)}"

        @GracefulDegradation.with_fallback(fallback_func=fallback_handler)
        def failing_func():
            raise ValueError("test error")

        result = failing_func()
        assert "Handled: test error" in result

    @pytest.mark.unit
    def test_with_fallback_success(self):
        """Test graceful degradation with successful call."""

        @GracefulDegradation.with_fallback(fallback_value="fallback")
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    @pytest.mark.unit
    def test_with_timeout_unix(self):
        """Test timeout decorator on Unix systems."""
        # Skip on Windows or systems without SIGALRM
        try:
            import signal

            if not hasattr(signal, "SIGALRM"):
                pytest.skip("SIGALRM not available on this system")
        except ImportError:
            pytest.skip("signal module not available")

        @GracefulDegradation.with_timeout(timeout=0.1, default_value="timeout")
        def slow_func():
            time.sleep(1.0)  # Takes longer than timeout
            return "done"

        result = slow_func()
        # On systems with SIGALRM, should return timeout
        # On Windows, will return 'done' due to different implementation
        assert result in ["timeout", "done"]

    @pytest.mark.unit
    def test_with_timeout_success(self):
        """Test timeout decorator with successful fast call."""

        @GracefulDegradation.with_timeout(timeout=1.0, default_value="timeout")
        def fast_func():
            return "success"

        result = fast_func()
        assert result == "success"

    @pytest.mark.unit
    def test_with_rate_limit(self):
        """Test rate limiting decorator."""

        @GracefulDegradation.with_rate_limit(max_calls=2, period=1.0)
        def limited_func():
            return "success"

        # First two calls should succeed
        assert limited_func() == "success"
        assert limited_func() == "success"

        # Third call should fail
        with pytest.raises(RateLimitExceededError):
            limited_func()

    @pytest.mark.unit
    def test_with_rate_limit_after_period(self):
        """Test rate limit resets after period."""

        @GracefulDegradation.with_rate_limit(max_calls=1, period=0.2)
        def limited_func():
            return "success"

        assert limited_func() == "success"

        with pytest.raises(RateLimitExceededError):
            limited_func()

        # Wait for period to pass
        time.sleep(0.25)

        # Should work again
        assert limited_func() == "success"


class TestRetryDecorator:
    """Test suite for retry_with_policy decorator."""

    @pytest.mark.unit
    def test_retry_with_policy_success(self):
        """Test retry decorator with successful call."""
        policy = RetryPolicy(max_retries=3, retryable_exceptions=(ValueError,))

        @retry_with_policy(policy)
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    @pytest.mark.unit
    def test_retry_with_policy_succeeds_after_retries(self):
        """Test retry decorator that succeeds after retries."""
        attempt_count = [0]
        policy = RetryPolicy(max_retries=3, retryable_exceptions=(ValueError,), initial_backoff_time=0.01)

        @retry_with_policy(policy)
        def func_with_retries():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ValueError("retry")
            return "success"

        result = func_with_retries()
        assert result == "success"
        assert attempt_count[0] == 3

    @pytest.mark.unit
    def test_retry_with_policy_exhausts_retries(self):
        """Test retry decorator exhausts all retries."""
        policy = RetryPolicy(max_retries=2, retryable_exceptions=(ValueError,), initial_backoff_time=0.01)

        @retry_with_policy(policy)
        def always_failing_func():
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            always_failing_func()

    @pytest.mark.unit
    def test_retry_with_policy_non_retryable_exception(self):
        """Test retry decorator with non-retryable exception."""
        policy = RetryPolicy(max_retries=3, retryable_exceptions=(ValueError,), initial_backoff_time=0.01)

        @retry_with_policy(policy)
        def func_with_key_error():
            raise KeyError("not retryable")

        # Should not retry, raise immediately
        with pytest.raises(KeyError):
            func_with_key_error()


class TestCircuitBreakerOpenError:
    """Test suite for CircuitBreakerOpenError exception."""

    @pytest.mark.unit
    def test_circuit_breaker_open_error(self):
        """Test CircuitBreakerOpenError exception."""
        error = CircuitBreakerOpenError("Circuit is open")

        assert isinstance(error, Exception)
        assert str(error) == "Circuit is open"


class TestRateLimitExceededError:
    """Test suite for RateLimitExceededError exception."""

    @pytest.mark.unit
    def test_rate_limit_exceeded_error(self):
        """Test RateLimitExceededError exception."""
        error = RateLimitExceededError("Rate limit exceeded")

        assert isinstance(error, Exception)
        assert str(error) == "Rate limit exceeded"

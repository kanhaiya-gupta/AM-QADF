"""
Fault tolerance and error recovery utilities.

This module provides error handling, retry logic, circuit breakers,
and graceful degradation capabilities.
"""

import time
import functools
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Tuple, List, Dict
from enum import Enum
from threading import Lock
from collections import deque


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryPolicy:
    """Retry policy configuration."""

    max_retries: int = 3
    backoff_factor: float = 1.0
    exponential_backoff: bool = True
    retryable_exceptions: Tuple = field(default_factory=lambda: (Exception,))
    max_backoff_time: float = 60.0  # Maximum backoff time in seconds
    initial_backoff_time: float = 1.0  # Initial backoff time in seconds

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """
        Determine if operation should be retried.

        Args:
            attempt: Current attempt number (0-indexed)
            exception: Exception that occurred

        Returns:
            True if operation should be retried
        """
        if attempt >= self.max_retries:
            return False

        # Check if exception is retryable
        if not isinstance(exception, self.retryable_exceptions):
            return False

        return True

    def get_backoff_time(self, attempt: int) -> float:
        """
        Calculate backoff time for retry.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Backoff time in seconds
        """
        if self.exponential_backoff:
            backoff = self.initial_backoff_time * (self.backoff_factor**attempt)
        else:
            backoff = self.initial_backoff_time * self.backoff_factor * (attempt + 1)

        return min(backoff, self.max_backoff_time)


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self, failure_threshold: int = 5, timeout: float = 60.0, half_open_max_calls: int = 3, success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time in seconds before attempting half-open state
            half_open_max_calls: Maximum calls allowed in half-open state
            success_threshold: Number of successes needed to close circuit from half-open
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time: Optional[float] = None
        self.lock = Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception raised by the function
        """
        with self.lock:
            # Check if circuit is open and timeout has passed
            if self.state == CircuitState.OPEN:
                if self.last_failure_time and (time.time() - self.last_failure_time) >= self.timeout:
                    # Transition to half-open
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}")

            # Check half-open call limit
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    # Too many calls in half-open, open circuit again
                    self.state = CircuitState.OPEN
                    self.last_failure_time = time.time()
                    raise CircuitBreakerOpenError("Circuit breaker exceeded half-open call limit")

        # Execute function
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise

    def record_success(self):
        """Record successful operation."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                self.half_open_calls += 1
                if self.success_count >= self.success_threshold:
                    # Close circuit
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.half_open_calls = 0
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    def record_failure(self):
        """Record failed operation."""
        with self.lock:
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                # Failure in half-open, open circuit
                self.state = CircuitState.OPEN
                self.failure_count = 0
                self.success_count = 0
                self.half_open_calls = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    # Open circuit
                    self.state = CircuitState.OPEN

    @property
    def current_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state.value

    def reset(self):
        """Reset circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            self.last_failure_time = None


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class GracefulDegradation:
    """Graceful degradation utilities."""

    @staticmethod
    def with_fallback(fallback_value: Any = None, fallback_func: Optional[Callable] = None):
        """
        Decorator for graceful degradation with fallback.

        Args:
            fallback_value: Default value to return on failure
            fallback_func: Function to call on failure (takes exception as argument)

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if fallback_func:
                        return fallback_func(e)
                    return fallback_value

            return wrapper

        return decorator

    @staticmethod
    def with_timeout(timeout: float, default_value: Any = None):
        """
        Decorator for operation timeout.

        Args:
            timeout: Timeout in seconds
            default_value: Value to return on timeout

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Operation timed out after {timeout} seconds")

                # Set up signal handler (Unix only)
                if hasattr(signal, "SIGALRM"):
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout))
                    try:
                        result = func(*args, **kwargs)
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                    return result
                else:
                    # Windows: Use threading approach
                    import threading

                    result_container = [None]
                    exception_container = [None]

                    def target():
                        try:
                            result_container[0] = func(*args, **kwargs)
                        except Exception as e:
                            exception_container[0] = e

                    thread = threading.Thread(target=target)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout)

                    if thread.is_alive():
                        # Thread still running, timeout occurred
                        return default_value

                    if exception_container[0]:
                        raise exception_container[0]

                    return result_container[0]

            return wrapper

        return decorator

    @staticmethod
    def with_rate_limit(max_calls: int, period: float):
        """
        Decorator for rate limiting.

        Args:
            max_calls: Maximum number of calls allowed
            period: Time period in seconds

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            call_times: deque = deque()
            lock = Lock()

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with lock:
                    now = time.time()
                    # Remove old call times outside the period
                    while call_times and call_times[0] < now - period:
                        call_times.popleft()

                    if len(call_times) >= max_calls:
                        # Rate limit exceeded
                        oldest_call = call_times[0]
                        wait_time = period - (now - oldest_call)
                        if wait_time > 0:
                            raise RateLimitExceededError(f"Rate limit exceeded. Wait {wait_time:.2f} seconds.")

                    call_times.append(now)

                return func(*args, **kwargs)

            return wrapper

        return decorator


class RateLimitExceededError(Exception):
    """Exception raised when rate limit is exceeded."""

    pass


def retry_with_policy(policy: RetryPolicy):
    """
    Decorator for retrying operations with a retry policy.

    Args:
        policy: Retry policy configuration

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(policy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(attempt, e):
                        raise

                    if attempt < policy.max_retries:
                        backoff_time = policy.get_backoff_time(attempt)
                        time.sleep(backoff_time)

            # All retries exhausted
            raise last_exception

        return wrapper

    return decorator

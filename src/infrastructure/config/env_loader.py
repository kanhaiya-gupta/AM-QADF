"""
Environment Variable Loader

Loads environment variables from .env files for Docker and local development.
"""

import os
from pathlib import Path
from typing import Optional, Dict
from dotenv import load_dotenv


def find_env_file(env_name: str = "development") -> Optional[Path]:
    """
    Find .env file in project root.

    Args:
        env_name: Environment name (development, production, etc.)

    Returns:
        Path to .env file or None if not found
    """
    # Try different possible locations
    possible_paths = [
        Path.cwd() / f"{env_name}.env",
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent / f"{env_name}.env",
        Path(__file__).parent.parent.parent / ".env",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


def load_env_file(env_name: str = "development", override: bool = False) -> Dict[str, str]:
    """
    Load environment variables from .env file.

    Args:
        env_name: Environment name (development, production, etc.)
        override: If True, override existing environment variables

    Returns:
        Dictionary of loaded environment variables
    """
    env_file = find_env_file(env_name)

    if env_file is None:
        # Fallback: try to load from environment variables directly
        # This works in Docker where env vars are set by docker-compose
        return {}

    # Load .env file
    load_dotenv(env_file, override=override)

    # Return all loaded variables
    return dict(os.environ)


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get environment variable value.

    Args:
        key: Environment variable name
        default: Default value if not found
        required: If True, raise error if variable not found

    Returns:
        Environment variable value or default

    Raises:
        ValueError: If required variable is not found
    """
    value = os.getenv(key, default)

    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")

    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean environment variable.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Boolean value
    """
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """
    Get integer environment variable.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Integer value
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """
    Get float environment variable.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Float value
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default

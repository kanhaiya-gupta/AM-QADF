"""
Mock query clients for testing.
"""

from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class MockQueryResult:
    """Mock query result."""

    def __init__(
        self,
        points: np.ndarray = None,
        signals: Dict[str, np.ndarray] = None,
        metadata: Dict[str, Any] = None,
    ):
        self.points = points if points is not None else np.array([], dtype=np.float64).reshape(0, 3)
        self.signals = signals if signals is not None else {}
        self.metadata = metadata if metadata is not None else {}


class MockSTLClient:
    """Mock STL model client."""

    def __init__(self):
        self.get_model_bounding_box = Mock(return_value=((-50, -50, -50), (50, 50, 50)))
        self.get_model_metadata = Mock(return_value={"model_id": "test_model", "volume": 1000.0})
        self.query = Mock(return_value=MockQueryResult())


class MockHatchingClient:
    """Mock hatching client."""

    def __init__(self):
        self.get_layers = Mock(return_value=[])
        self.get_hatch_paths = Mock(return_value=[])
        self.get_all_points = Mock(return_value=np.array([], dtype=np.float64).reshape(0, 3))
        self.query = Mock(return_value=MockQueryResult())
        self.query_spatial = Mock(return_value=[])
        self.query_temporal = Mock(return_value=[])


class MockLaserClient:
    """Mock laser parameter client."""

    def __init__(self):
        self.query = Mock(
            return_value=MockQueryResult(
                points=np.random.rand(100, 3) * 100.0,
                signals={"laser_power": np.random.rand(100) * 300.0},
            )
        )
        self.query_spatial = Mock(return_value=MockQueryResult())
        self.query_temporal = Mock(return_value=MockQueryResult())
        self.query_by_signal_types = Mock(return_value=MockQueryResult())


class MockCTClient:
    """Mock CT scan client."""

    def __init__(self):
        self.query = Mock(
            return_value=MockQueryResult(
                points=np.random.rand(50, 3) * 100.0,
                signals={"density": np.random.rand(50) * 1.0 + 4.0},
            )
        )
        self.query_spatial = Mock(return_value=MockQueryResult())
        self.query_temporal = Mock(return_value=MockQueryResult())
        self.get_voxel_grid_metadata = Mock(
            return_value={
                "dimensions": [100, 100, 100],
                "spacing": [0.1, 0.1, 0.1],
                "origin": [0.0, 0.0, 0.0],
            }
        )


class MockISPMClient:
    """Mock in-situ process monitoring client."""

    def __init__(self):
        self.query = Mock(
            return_value=MockQueryResult(
                points=np.random.rand(200, 3) * 100.0,
                signals={"temperature": np.random.rand(200) * 1000.0},
            )
        )
        self.query_spatial = Mock(return_value=MockQueryResult())
        self.query_temporal = Mock(return_value=MockQueryResult())
        self.query_by_signal_types = Mock(return_value=MockQueryResult())


class MockUnifiedQueryClient:
    """Mock unified query client."""

    def __init__(self, mongo_client=None):
        self.mongo_client = mongo_client

        # Initialize sub-clients
        self.stl_client = MockSTLClient()
        self.hatching_client = MockHatchingClient()
        self.laser_client = MockLaserClient()
        self.ct_client = MockCTClient()
        self.ispm_client = MockISPMClient()

        # Mock coordinate transformer
        self.get_coordinate_transformer = Mock(return_value=None)

        # Mock query methods
        self.query = Mock(
            return_value={
                "hatching": MockQueryResult(),
                "laser": MockQueryResult(),
                "ct": MockQueryResult(),
                "ispm": MockQueryResult(),
            }
        )

        self.query_all_sources = Mock(
            return_value={
                "hatching": MockQueryResult(),
                "laser": MockQueryResult(),
                "ct": MockQueryResult(),
                "ispm": MockQueryResult(),
            }
        )

        self.query_spatial = Mock(
            return_value={
                "hatching": MockQueryResult(),
                "laser": MockQueryResult(),
                "ct": MockQueryResult(),
                "ispm": MockQueryResult(),
            }
        )

        self.query_temporal = Mock(
            return_value={
                "hatching": MockQueryResult(),
                "laser": MockQueryResult(),
                "ct": MockQueryResult(),
                "ispm": MockQueryResult(),
            }
        )


def create_mock_unified_query_client(mongo_client=None) -> MockUnifiedQueryClient:
    """Create a mock unified query client."""
    return MockUnifiedQueryClient(mongo_client=mongo_client)

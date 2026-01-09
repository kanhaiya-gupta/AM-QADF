# Test Structure

## Directory Organization

```
tests/
├── __init__.py
├── conftest.py                    # Shared pytest fixtures
│
├── unit/                          # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── core/                      # Core module tests
│   ├── query/                     # Query module tests
│   ├── voxelization/              # Voxelization tests
│   ├── signal_mapping/            # Signal mapping tests
│   ├── synchronization/           # Synchronization tests
│   ├── correction/                # Correction tests
│   ├── processing/                # Processing tests
│   ├── fusion/                    # Fusion tests
│   ├── quality/                   # Quality tests
│   ├── analytics/                 # Analytics tests
│   ├── anomaly_detection/         # Anomaly detection tests
│   ├── visualization/             # Visualization tests
│   └── voxel_domain/              # Voxel domain tests
│
├── integration/                   # Integration tests
│   ├── test_signal_mapping_pipeline.py
│   ├── test_voxel_domain_workflow.py
│   ├── test_analytics_workflow.py
│   ├── test_fusion_workflow.py
│   ├── test_quality_assessment_workflow.py
│   └── test_end_to_end_workflow.py
│
├── performance/                   # Performance tests
│   ├── benchmarks/                # Benchmark tests
│   │   ├── benchmark_signal_mapping.py
│   │   ├── benchmark_voxel_fusion.py
│   │   ├── benchmark_interpolation_methods.py
│   │   └── benchmark_parallel_execution.py
│   └── regression/                # Regression tests
│       ├── test_performance_regression.py
│       └── test_memory_regression.py
│
├── fixtures/                      # Test data and fixtures
│   ├── voxel_data/                # Voxel grid fixtures
│   ├── point_clouds/              # Point cloud fixtures
│   ├── signals/                   # Signal fixtures
│   └── mocks/                     # Mock objects
│
├── property_based/                # Property-based tests
│   ├── test_voxel_grid_properties.py
│   ├── test_interpolation_properties.py
│   ├── test_fusion_properties.py
│   └── test_coordinate_transformations.py
│
└── e2e/                           # End-to-end tests
    ├── test_complete_pipeline.py
    ├── test_multi_source_fusion.py
    └── test_analytics_pipeline.py
```

## Naming Conventions

### Test Files
- Pattern: `test_<module_name>.py`
- Example: `test_voxel_grid.py`, `test_hatching_client.py`

### Test Functions
- Pattern: `test_<functionality>_<condition>`
- Example: `test_interpolation_with_empty_data()`
- Example: `test_fusion_commutativity()`

### Test Classes
- Pattern: `Test<ClassName>`
- Example: `TestVoxelGrid`, `TestInterpolationMethod`

## File Organization

### One Test File Per Source File
- Mirror source directory structure
- `src/am_qadf/voxelization/voxel_grid.py` → `tests/unit/voxelization/test_voxel_grid.py`

### Group Related Tests
- Use test classes to group related tests
- Use pytest markers for categorization

## Test File Structure Template

```python
"""
Unit tests for <ModuleName>.

Tests for <brief description>.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

try:
    from am_qadf.<module> import <Class>
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False


@pytest.mark.skipif(not MODULE_AVAILABLE, reason="Module not available")
@pytest.mark.unit
class TestClassName:
    """Test cases for ClassName."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return ...
    
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        # Arrange
        obj = ClassName(...)
        
        # Act
        result = obj.method(sample_data)
        
        # Assert
        assert result is not None
```

## Related Documentation

- [Module Testing Guides](05-module-testing/) - Module-specific structure
- [Test Categories](04-test-categories/) - Category-specific organization
- [Infrastructure](06-infrastructure.md) - Fixtures and utilities

---

**Related**: [Overview](01-overview.md) | [Test Categories](04-test-categories/)


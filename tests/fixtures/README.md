# Test Fixtures

This directory contains pre-computed test data and fixtures for use in AM-QADF tests.

## Directory Structure

```
fixtures/
├── __init__.py              # Main fixtures module with pytest fixtures
├── README.md                # This file
├── voxel_data/              # Voxel grid fixtures
│   ├── __init__.py          # Voxel data loading utilities
│   ├── small_voxel_grid.pkl # Small voxel grid (10x10x10)
│   ├── medium_voxel_grid.pkl # Medium voxel grid (50x50x50)
│   └── large_voxel_grid.pkl  # Large voxel grid (100x100x100)
├── point_clouds/            # Point cloud data fixtures
│   ├── __init__.py          # Point cloud loading utilities
│   ├── hatching_paths.json  # Hatching path data
│   ├── laser_points.json    # Laser parameter points
│   └── ct_points.json       # CT scan points
├── signals/                 # Signal array fixtures
│   ├── __init__.py          # Signal loading utilities
│   └── sample_signals.npz   # Sample signal arrays
└── mocks/                   # Mock objects and utilities
    ├── __init__.py          # Mock exports
    ├── mock_mongodb.py      # MongoDB mock client
    ├── mock_spark.py        # Spark session mocks
    └── mock_query_clients.py  # Query client mocks
```

**Note**: Fixture generation scripts are located in `scripts/generate_test_fixtures.py`

## Generating Fixtures

To generate all fixture files, run:

```bash
# Generate all fixtures (voxel grids, point clouds, signals)
python scripts/generate_test_fixtures.py
```

This will create:
- Pickle files for small, medium, and large voxel grids in `tests/fixtures/voxel_data/`
- JSON files for hatching paths, laser points, and CT scan points in `tests/fixtures/point_clouds/`
- NPZ file for sample signal arrays in `tests/fixtures/signals/`

## Using Fixtures in Tests

### Using Pytest Fixtures

The easiest way to use fixtures in tests is via pytest fixtures:

```python
def test_something(small_voxel_grid):
    """Test using small voxel grid fixture."""
    assert small_voxel_grid is not None
    assert len(small_voxel_grid.available_signals) > 0
```

### Direct Loading

You can also load fixtures directly:

```python
from tests.fixtures.voxel_data import load_small_voxel_grid

def test_something():
    grid = load_small_voxel_grid()
    # Use grid...
```

### On-the-Fly Generation

If fixture files don't exist, the loading functions will automatically generate them on-the-fly:

```python
from tests.fixtures.voxel_data import load_medium_voxel_grid

# This will generate the fixture if it doesn't exist
grid = load_medium_voxel_grid()
```

## Available Fixtures

### Voxel Grids

- **Small Voxel Grid** (`small_voxel_grid`):
  - Dimensions: 10x10x10 voxels
  - Bounding box: (0, 0, 0) to (10, 10, 10) mm
  - Resolution: 1.0 mm
  - Signals: `laser_power`, `temperature`
  - Points: ~100 data points

- **Medium Voxel Grid** (`medium_voxel_grid`):
  - Dimensions: 50x50x50 voxels
  - Bounding box: (0, 0, 0) to (50, 50, 50) mm
  - Resolution: 1.0 mm
  - Signals: `laser_power`, `temperature`, `density`
  - Points: ~1000 data points

- **Large Voxel Grid** (`large_voxel_grid`):
  - Dimensions: 100x100x100 voxels
  - Bounding box: (0, 0, 0) to (100, 100, 100) mm
  - Resolution: 1.0 mm
  - Signals: `laser_power`, `temperature`, `density`, `velocity`
  - Points: ~5000 data points

## Fixture Properties

All voxel grid fixtures:
- Use a fixed random seed (42) for reproducibility
- Have signals aggregated using the default 'mean' method
- Are finalized (ready to use) when loaded
- Include metadata about available signals

## Best Practices

1. **Use appropriate fixture size**: Use small grids for unit tests, medium for integration tests, and large for performance tests.

2. **Don't modify fixtures**: Fixtures should be treated as read-only. If you need to modify data, create a copy first.

3. **Regenerate when needed**: If you change the fixture generation logic, regenerate all fixtures to ensure consistency.

4. **Version control**: Commit fixture files to version control so all developers have the same test data.

## Available Mock Objects

### MongoDB Mocks

```python
from tests.fixtures.mocks import MockMongoClient, MockCollection

# Create a mock MongoDB client
mock_client = MockMongoClient()
collection = mock_client.get_collection('test_collection')
```

### Spark Mocks

```python
from tests.fixtures.mocks import MockSparkSession, MockDataFrame

# Create a mock Spark session
spark = MockSparkSession()
df = spark.createDataFrame([{'x': 1, 'y': 2}])
```

### Query Client Mocks

```python
from tests.fixtures.mocks import MockUnifiedQueryClient

# Create a mock unified query client
client = MockUnifiedQueryClient()
result = client.query(model_id='test_model')
```

## Point Cloud Fixtures

### Hatching Paths

```python
from tests.fixtures.point_clouds import load_hatching_paths

layers = load_hatching_paths()
# Returns list of layer dictionaries with hatches
```

### Laser Points

```python
from tests.fixtures.point_clouds import load_laser_points

points = load_laser_points()
# Returns list of laser parameter point dictionaries
```

### CT Scan Points

```python
from tests.fixtures.point_clouds import load_ct_points

ct_data = load_ct_points()
# Returns CT scan data dictionary with points and metadata
```

## Signal Fixtures

```python
from tests.fixtures.signals import load_sample_signals

signals = load_sample_signals()
# Returns dictionary with signal arrays:
# - laser_power, scan_speed, temperature
# - density, porosity, velocity
# - energy_density, exposure_time
```

## Adding New Fixtures

To add a new fixture:

1. Create a function in the appropriate `__init__.py` file
2. Add a pytest fixture in `tests/fixtures/__init__.py`
3. Update the generation script if needed
4. Document the fixture in this README


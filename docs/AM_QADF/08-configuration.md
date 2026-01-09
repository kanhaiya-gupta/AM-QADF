# Configuration Guide

## Configuration Options

### MongoDB Configuration

```python
from am_qadf.query.mongodb_client import MongoDBClient

mongodb_client = MongoDBClient(
    connection_string="mongodb://localhost:27017",
    database_name="am_qadf"
)
```

### Voxel Grid Configuration

```python
from am_qadf.voxelization import VoxelGrid

grid = VoxelGrid(
    bbox_min=(0, 0, 0),
    bbox_max=(100, 100, 100),
    resolution=1.0  # mm
)
```

### Signal Mapping Configuration

```python
# Interpolation method
method = 'nearest'  # or 'linear', 'idw', 'kde'

# Parallel execution
n_workers = 4
```

## Environment Variables

```bash
# MongoDB connection
export MONGODB_URI="mongodb://localhost:27017"

# Spark configuration
export SPARK_HOME="/path/to/spark"
```

## Related

- [Installation](03-installation.md) - Installation guide
- [Modules](05-modules/README.md) - Module-specific configuration

---

**Parent**: [Framework Documentation](README.md)



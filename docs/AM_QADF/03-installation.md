# Installation Guide

## Requirements

- Python 3.9 or higher
- NumPy
- SciPy
- Pandas
- MongoDB (for storage)
- Optional: Apache Spark (for distributed processing)

## Installation

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd AM-QADF

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install testing dependencies
pip install -r requirements-test.txt
```

## Dependencies

### Core Dependencies
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `pandas` - Data manipulation
- `pymongo` - MongoDB client

### Optional Dependencies
- `pyspark` - Apache Spark (for distributed processing)
- `tensorflow` - Machine learning (for ML-based anomaly detection)
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Plotting
- `pyvista` - 3D visualization

## MongoDB Setup

AM-QADF uses MongoDB for data storage. Set up MongoDB:

```bash
# Using Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Or install MongoDB locally
# See: https://www.mongodb.com/docs/manual/installation/
```

## Verification

Verify installation:

```python
import am_qadf
print(am_qadf.__version__)

# Test imports
from am_qadf.voxelization import VoxelGrid
from am_qadf.query import UnifiedQueryClient
print("Installation successful!")
```

## Next Steps

- **[Quick Start](04-quick-start.md)** - Get started with examples
- **[Configuration](08-configuration.md)** - Configure the framework

---

**Related**: [Quick Start](04-quick-start.md) | [Configuration](08-configuration.md)


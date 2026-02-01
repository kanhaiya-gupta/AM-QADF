# Installation Guide

AM-QADF has a **Python** package and an optional **C++ native extension** (`am_qadf_native`). You can install Python-only for basic use; for full functionality (synchronization, voxelization, fusion, advanced query), the C++ extension is built as part of the package install. See [Python and C++](12-python-and-cpp.md).

## Requirements

### Python
- Python 3.8 or higher (3.9+ recommended)
- NumPy, SciPy, Pandas
- MongoDB client (pymongo) for storage/query

### C++ extension (built automatically when possible)
- CMake ≥ 3.15, C++17 compiler, Ninja
- pybind11 (supplied via build)
- OpenVDB, Eigen (and optionally ITK, mongocxx, TBB) for full native features

If these are not available, `pip install` still installs the Python package; some features will require the native module and will report it when used.

## Installation

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd AM-QADF

# Install dependencies
pip install -r requirements.txt

# Install the package (builds C++ extension if CMake and dependencies are available)
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

### Core Python Dependencies
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `pandas` - Data manipulation
- `pymongo` - MongoDB client

### Build (for C++ extension)
- `cmake`, `pybind11`, `ninja` - Used by the build system to compile `am_qadf_native`

### Optional Python Dependencies
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

# Test Python imports
from am_qadf.voxelization import VoxelGrid
from am_qadf.query import UnifiedQueryClient
print("Python package OK")

# Optional: check C++ extension (needed for full pipeline)
try:
    import am_qadf_native
    print("C++ extension (am_qadf_native) is available")
except ImportError:
    print("C++ extension not built; some features will be limited (see Python and C++ doc)")
```

## Next Steps

- **[Quick Start](04-quick-start.md)** - Get started with examples
- **[Configuration](08-configuration.md)** - Configure the framework

---

**Related**: [Quick Start](04-quick-start.md) | [Configuration](08-configuration.md)


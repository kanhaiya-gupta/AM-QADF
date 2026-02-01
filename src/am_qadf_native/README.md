# AM-QADF Native C++ Module

This directory contains the C++ core implementation of AM-QADF using OpenVDB, ITK, and mongocxx.

## Structure

```
am_qadf_native/
├── include/am_qadf_native/    # Public API headers
│   ├── voxelization/          # Voxel grid implementations
│   ├── signal_mapping/        # Signal mapping/interpolation
│   ├── fusion/                # Grid fusion operations
│   ├── synchronization/       # Spatial/temporal alignment
│   ├── query/                 # MongoDB queries, CT scan reading
│   ├── correction/            # Noise reduction, geometric correction
│   ├── processing/            # Signal processing and generation
│   └── io/                    # OpenVDB I/O, ParaView export
├── src/                       # Implementation files
│   └── [same structure as include]
└── python/                    # Python bindings (pybind11)
    └── bindings/              # Binding files
```

## Building

### Prerequisites

- CMake >= 3.15
- C++17 compiler
- OpenVDB (system installation recommended)
- ITK (for CT scan reading)
- mongocxx (for MongoDB queries)
- pybind11 (for Python bindings)

### Build Steps

```bash
mkdir build
cd build
cmake ..
make
```

## Status

**Phase 0**: ✅ Complete - All files created with basic structure
- All header files (.hpp) created
- All source files (.cpp) created with implementations
- Python bindings created
- CMakeLists.txt configured

**Next Steps**: Implement Phase 1 (OpenVDB Integration Foundation)

## Files Created

### Headers (27 files)
- Voxelization: 3 headers
- Signal Mapping: 6 headers
- Fusion: 2 headers
- Synchronization: 3 headers
- Query: 6 headers
- Correction: 5 headers
- Processing: 2 headers
- I/O: 3 headers

### Sources (27 files)
- Corresponding .cpp files for all headers

### Python Bindings (8 files)
- Module bindings for all components

### Build System
- CMakeLists.txt (root and module)
- FindOpenVDB.cmake
- FindITK.cmake
- pyproject.toml
- setup.py

## Implementation Status

Most files have basic implementations. Some complex algorithms (IDW, KDE, RBF, ITK integration) have placeholder implementations marked with `// TODO:` comments.

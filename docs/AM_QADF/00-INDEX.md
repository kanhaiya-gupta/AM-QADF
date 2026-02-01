# AM-QADF Framework Documentation Index

## Quick Start

1. **[Overview](01-overview.md)** - Framework introduction
2. **[Installation](03-installation.md)** - Setup instructions
3. **[Quick Start](04-quick-start.md)** - Get started quickly
4. **[Architecture](02-architecture.md)** - System design

## Complete Documentation

### Getting Started
- [01. Overview](01-overview.md) - Framework goals and capabilities
- [02. Architecture](02-architecture.md) - System architecture and design patterns
- [03. Installation](03-installation.md) - Installation and dependencies
- [04. Quick Start](04-quick-start.md) - Quick start guide with examples

### Module Documentation
- [05. Modules](05-modules/README.md) - All framework modules
  - [Core](05-modules/core.md) - Domain entities, value objects, exceptions
  - [Query](05-modules/query.md) - Data warehouse query clients
  - [Voxelization](05-modules/voxelization.md) - Voxel grid operations
  - [Signal Mapping](05-modules/signal-mapping.md) - ⭐ **CRITICAL** - Interpolation methods
  - [Synchronization](05-modules/synchronization.md) - Temporal/spatial alignment
  - [Correction](05-modules/correction.md) - Geometric distortion correction
  - [Processing](05-modules/processing.md) - Signal processing
  - [Fusion](05-modules/fusion.md) - Multi-source data fusion
  - [Quality](05-modules/quality.md) - Quality assessment
  - [Validation](05-modules/validation.md) - Validation and benchmarking
  - [Analytics](05-modules/analytics.md) - Statistical and sensitivity analysis
  - [SPC](05-modules/spc.md) - Statistical Process Control (control charts, capability, multivariate SPC)
  - [Anomaly Detection](05-modules/anomaly-detection.md) - Anomaly detection
  - [Visualization](05-modules/visualization.md) - 3D visualization
  - [Voxel Domain](05-modules/voxel-domain.md) - Main orchestrator

### Reference Documentation
- [06. API Reference](06-api-reference/README.md) - Complete API documentation
- [07. Examples](07-examples/README.md) - Usage examples and tutorials
- [08. Configuration](08-configuration.md) - Configuration options
- [09. Performance](09-performance.md) - Performance considerations

### Implementation
- [12. Python and C++](12-python-and-cpp.md) - Python API and C++ native extension

### Support
- [10. Troubleshooting](10-troubleshooting.md) - Common issues and solutions
- [11. Contributing](11-contributing.md) - Contributing guidelines

## Module Quick Reference

| Module | Purpose | Key Components |
|--------|---------|----------------|
| **Core** | Domain foundation | Entities, Value Objects, Exceptions |
| **Query** | Data access | Query clients for all data sources |
| **Voxelization** | Spatial discretization | VoxelGrid, CoordinateSystems |
| **Signal Mapping** | Interpolation | Nearest, Linear, IDW, KDE methods |
| **Synchronization** | Alignment | Temporal, Spatial transformations |
| **Correction** | Calibration | Geometric distortion correction |
| **Processing** | Signal processing | Noise reduction, signal generation |
| **Fusion** | Data fusion | Multi-source fusion strategies |
| **Quality** | Quality assessment | Completeness, signal quality metrics |
| **Validation** | Validation & benchmarking | Performance, MPM comparison, accuracy, statistical tests |
| **Analytics** | Analysis | Statistical, sensitivity analysis |
| **SPC** | Process control | Control charts, capability analysis, multivariate SPC, control rules |
| **Anomaly Detection** | Anomaly detection | Multiple detection algorithms |
| **Visualization** | 3D visualization | Rendering, widgets, viewers |
| **Voxel Domain** | Orchestration | Main client, storage |

---

**Last Updated**: 2024


# AM-QADF Framework Documentation

**Version**: 1.0  
**Last Updated**: 2024  
**Status**: 📚 Documentation

## Overview

This directory contains comprehensive, modular documentation for the AM-QADF (Additive Manufacturing Quality Assessment and Data Fusion) framework. The documentation is organized into focused modules for easy navigation and maintenance.

## Documentation Structure

```
docs/AM_QADF/
├── README.md                    # This file - navigation guide
├── 00-INDEX.md                  # Complete documentation index
├── 01-overview.md               # Framework overview and goals
├── 02-architecture.md            # System architecture and design
├── 03-installation.md            # Installation and setup
├── 04-quick-start.md            # Quick start guide
├── 05-modules/                   # Module-specific documentation
│   ├── README.md
│   ├── core.md
│   ├── query.md
│   ├── voxelization.md
│   ├── signal-mapping.md ⭐
│   ├── synchronization.md
│   ├── correction.md
│   ├── processing.md
│   ├── fusion.md
│   ├── quality.md
│   ├── analytics.md
│   ├── anomaly-detection.md
│   ├── visualization.md
│   └── voxel-domain.md
├── 06-api-reference/             # API documentation
│   ├── README.md
│   └── [module-specific API docs]
├── 07-examples/                  # Usage examples
│   ├── README.md
│   └── [example notebooks/scripts]
├── 08-configuration.md           # Configuration guide
├── 09-performance.md             # Performance considerations
├── 10-troubleshooting.md         # Common issues and solutions
└── 11-contributing.md            # Contributing guidelines
```

## Quick Navigation

### Getting Started
- **[Overview](01-overview.md)** - Start here for framework introduction
- **[Installation](03-installation.md)** - Setup instructions
- **[Quick Start](04-quick-start.md)** - Get up and running quickly
- **[Architecture](02-architecture.md)** - Understand the system design

### Module Documentation
- **[Modules](05-modules/README.md)** - All framework modules
  - [Core](05-modules/core.md) - Domain entities and value objects
  - [Query](05-modules/query.md) - Data warehouse query clients
  - [Voxelization](05-modules/voxelization.md) - Voxel grid operations
  - [Signal Mapping](05-modules/signal-mapping.md) - ⭐ **CRITICAL
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

### Reference
- **[API Reference](06-api-reference/)** - Complete API documentation
- **[Examples](07-examples/)** - Usage examples and tutorials
- **[Configuration](08-configuration.md)** - Configuration options
- **[Performance](09-performance.md)** - Performance optimization

### Support
- **[Troubleshooting](10-troubleshooting.md)** - Common issues and solutions
- **[Contributing](11-contributing.md)** - How to contribute

## Framework Overview

AM-QADF is a comprehensive framework for:
- **Data Fusion**: Combining multi-source additive manufacturing data
- **Quality Assessment**: Evaluating data quality and completeness
- **Analytics**: Statistical and sensitivity analysis
- **Anomaly Detection**: Identifying anomalies in manufacturing data
- **Visualization**: 3D visualization of voxel domain data

## Related Documentation

- **[Testing Documentation](../Tests/README.md)** - Comprehensive testing guide
- **[Testing Plan](../TESTING_PLAN.md)** - Testing strategy

---

**Last Updated**: 2024


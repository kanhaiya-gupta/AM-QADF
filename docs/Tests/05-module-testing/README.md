# Module-Specific Testing Guides

This directory contains testing guides for each module in the AM-QADF framework.

## Modules

1. **[Core](core.md)** - Domain entities, value objects, exceptions
2. **[Query](query.md)** - Query clients for data warehouse
3. **[Voxelization](voxelization.md)** - Voxel grid operations
4. **[Signal Mapping](signal-mapping.md)** - ⭐ **CRITICAL** - Interpolation methods
5. **[Synchronization](synchronization.md)** - Temporal/spatial alignment, fusion
6. **[Correction](correction.md)** - Geometric distortion, calibration
7. **[Processing](processing.md)** - Noise reduction, signal generation
8. **[Fusion](fusion.md)** - Voxel-level fusion strategies
9. **[Quality](quality.md)** - Quality metrics and assessment
10. **[Analytics](analytics.md)** - Statistical and sensitivity analysis
11. **[Anomaly Detection](anomaly-detection.md)** - Anomaly detection algorithms
12. **[Visualization](visualization.md)** - 3D rendering and widgets
13. **[Voxel Domain](voxel-domain.md)** - Main orchestrator and storage

## Quick Reference

| Module | Test Files | Coverage Target | Priority |
|--------|------------|----------------|----------|
| Core | 3 | 95% | High |
| Query | 9 | 85% | High |
| Voxelization | 5 | 90% | High |
| Signal Mapping | 13 | **95%** | **Critical** |
| Synchronization | 3 | 85% | Medium |
| Correction | 3 | 80% | Medium |
| Processing | 2 | 85% | Medium |
| Fusion | 3 | 90% | High |
| Quality | 4 | 85% | High |
| Analytics | 30+ | 80% | Medium |
| Anomaly Detection | 50+ | 80% | Medium |
| Visualization | 5 | 70% | Low |
| Voxel Domain | 2 | 85% | High |

## Navigation

Select a module above to view its specific testing guide.

---

**Parent**: [Test Documentation](../README.md)


# Coverage Requirements

## Coverage Targets

| Module | Target Coverage | Critical Paths |
|--------|----------------|---------------|
| `core/` | 95% | All entities, value objects |
| `query/` | 85% | Query construction, error handling |
| `voxelization/` | 90% | Voxel operations, transformations |
| `signal_mapping/` | **95%** | **All interpolation methods** |
| `synchronization/` | 85% | Alignment, fusion |
| `correction/` | 80% | Distortion models |
| `processing/` | 85% | Noise reduction, signal generation |
| `fusion/` | 90% | Fusion strategies |
| `quality/` | 85% | Quality metrics |
| `analytics/` | 80% | Analysis algorithms |
| `anomaly_detection/` | 80% | Detection algorithms |
| `visualization/` | 70% | Rendering (visual tests) |
| `voxel_domain/` | 85% | Orchestration, storage |

**Overall Target**: **80% minimum**, **85% preferred**

## Coverage Metrics

- **Code Coverage**: >80% overall, >95% for critical paths
- **Branch Coverage**: >75%
- **Function Coverage**: >85%

## Coverage Exclusions

- `__init__.py` files (unless they contain logic)
- Type stubs
- Deprecated code
- Platform-specific code (with platform markers)

## Measuring Coverage

```bash
# Run with coverage
pytest tests/ --cov=am_qadf --cov-report=term-missing

# Generate HTML report
pytest tests/ --cov=am_qadf --cov-report=html

# Check specific module
pytest tests/unit/voxelization/ --cov=am_qadf.voxelization
```

## Related

- [Success Metrics](12-success-metrics.md) - Coverage as a success metric
- [Best Practices](11-best-practices.md) - Coverage best practices

---

**Parent**: [Test Documentation](README.md)


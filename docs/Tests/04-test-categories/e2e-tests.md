# End-to-End Tests

## Purpose

Test complete workflows from data input to output.

## Characteristics

- **Speed**: 10-60 seconds per test
- **Scope**: Full framework pipeline
- **Dependencies**: Test database, realistic data
- **Count**: 3 test files

## Test Files

- `test_complete_pipeline.py` - Complete pipeline workflow
- `test_multi_source_fusion.py` - Multi-source fusion workflow
- `test_analytics_pipeline.py` - Analytics pipeline (query → analyze → report)

## Key Areas

- Complete signal mapping workflow
- Multi-source data fusion workflow
- Analytics pipeline (query → analyze → report)
- Quality assessment workflow

## Example Structure

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_complete_pipeline():
    """Test complete pipeline from query to output."""
    # Step 1: Query data
    query_client = UnifiedQueryClient(...)
    data = query_client.query_all_sources(model_id="test")
    
    # Step 2: Map signals
    voxel_client = VoxelDomainClient(...)
    grid = voxel_client.map_signals_to_voxels(...)
    
    # Step 3: Fuse data
    fusion_client = MultiVoxelGridFusion(...)
    fused = fusion_client.fuse_signals(...)
    
    # Step 4: Assess quality
    quality_client = QualityAssessmentClient(...)
    quality = quality_client.assess(...)
    
    # Assert
    assert fused is not None
    assert quality.score > 0.8
```

## Test Characteristics

- **Comprehensive**: Test full workflows
- **Realistic**: Use realistic test data
- **Slow**: Marked as `@pytest.mark.slow`
- **Integration**: Test multiple modules together

## Running E2E Tests

```bash
# Run all E2E tests
pytest tests/e2e/ -m e2e

# Run with verbose output
pytest tests/e2e/ -v -m e2e

# Run only slow tests (includes E2E)
pytest tests/e2e/ -m "e2e and slow"
```

## Best Practices

1. **Complete Workflows**: Test end-to-end scenarios
2. **Realistic Data**: Use comprehensive test data
3. **Error Handling**: Test error scenarios
4. **Isolation**: Each test should be independent
5. **Documentation**: Document complex workflows

## Related

- [Integration Tests](integration-tests.md) - Less comprehensive workflow tests
- [Module Testing Guides](../05-module-testing/) - Module-specific guides

---

**Parent**: [Test Categories](README.md)


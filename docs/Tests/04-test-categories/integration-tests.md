# Integration Tests

## Purpose

Test interactions between modules.

## Characteristics

- **Speed**: 1-10 seconds per test
- **Scope**: Multiple modules working together
- **Dependencies**: In-memory databases, test fixtures
- **Count**: 6 test files

## Key Areas

- Signal mapping pipeline (query → transform → voxelize → interpolate)
- Voxel domain workflow (multi-source data → fusion → storage)
- Analytics workflows (query → analyze → store results)
- Quality assessment pipeline

## Test Files

- `test_signal_mapping_pipeline.py` - Signal mapping workflow
- `test_voxel_domain_workflow.py` - Voxel domain operations
- `test_analytics_workflow.py` - Analytics workflows
- `test_fusion_workflow.py` - Fusion workflows
- `test_quality_assessment_workflow.py` - Quality assessment
- `test_end_to_end_workflow.py` - End-to-end integration

## Example Structure

```python
@pytest.mark.integration
def test_signal_mapping_pipeline():
    """Test complete signal mapping pipeline."""
    # Arrange
    query_client = MockQueryClient()
    voxel_grid = create_test_voxel_grid()
    
    # Act
    result = VoxelDomainClient(query_client).map_signals_to_voxels(
        model_id="test_model",
        signals=["laser_power", "temperature"]
    )
    
    # Assert
    assert result.has_signal("laser_power")
    assert result.has_signal("temperature")
```

## Best Practices

1. **Realistic Data**: Use realistic test data
2. **Isolation**: Each test should be independent
3. **Mocking**: Mock external services (MongoDB, etc.)
4. **Clear Workflows**: Test complete workflows
5. **Error Handling**: Test error scenarios

## Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -m integration

# Run specific test
pytest tests/integration/test_signal_mapping_pipeline.py
```

## Related

- [E2E Tests](e2e-tests.md) - More comprehensive workflow tests
- [Module Testing Guides](../05-module-testing/) - Module-specific guides

---

**Parent**: [Test Categories](README.md)



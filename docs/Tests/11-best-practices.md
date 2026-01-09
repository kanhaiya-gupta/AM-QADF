# Testing Best Practices

## Test Naming

- Use descriptive names: `test_nearest_neighbor_interpolation_with_empty_data`
- Follow pattern: `test_<functionality>_<condition>`
- Group related tests in classes

## Test Organization

- One test file per source file
- Mirror source directory structure
- Group related tests in classes

## Assertions

- Use specific assertions: `assert np.allclose()` not `assert result is not None`
- Include error messages: `assert condition, "Error message"`
- Test both positive and negative cases

## Test Independence

- Tests should not depend on execution order
- Use fixtures for setup/teardown
- Clean up after tests (delete temp files, etc.)

## Documentation

- Document complex test cases
- Explain why tests exist (regression tests)
- Include references to issues/PRs

## AAA Pattern

Always use Arrange-Act-Assert:

```python
def test_example():
    # Arrange
    data = create_test_data()
    obj = ClassName()
    
    # Act
    result = obj.method(data)
    
    # Assert
    assert result is not None
    assert result.value == expected_value
```

## Mocking

- Mock external dependencies
- Use `unittest.mock` or `pytest-mock`
- Verify mock calls when important

## Performance

- Keep unit tests fast (< 1s)
- Use appropriate test categories
- Mark slow tests with `@pytest.mark.slow`

## Related

- [Test Categories](04-test-categories/) - Category-specific best practices
- [Module Testing Guides](05-module-testing/) - Module-specific best practices

---

**Parent**: [Test Documentation](README.md)


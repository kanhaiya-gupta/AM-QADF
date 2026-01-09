# CI/CD Integration

## GitHub Actions Workflow

**`.github/workflows/tests.yml`**:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest tests/unit -v --cov
      
  integration-tests:
    runs-on: ubuntu-latest
    services:
      mongodb:
        image: mongo:latest
    steps:
      - uses: actions/checkout@v3
      - run: pytest tests/integration -v
      
  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pytest tests/performance -v --benchmark-only
```

## Pre-commit Hooks

- Run unit tests
- Check code coverage
- Lint code
- Format code (black, isort)

## Test Execution Strategy

### On Push/PR
- Run all unit tests
- Run integration tests
- Check coverage thresholds

### Scheduled (Nightly)
- Run full test suite
- Run performance benchmarks
- Generate coverage reports

### On Release
- Run all tests (unit, integration, E2E)
- Run performance regression tests
- Generate comprehensive reports

## Related

- [Success Metrics](12-success-metrics.md) - CI/CD success metrics
- [Best Practices](11-best-practices.md) - CI/CD best practices

---

**Parent**: [Test Documentation](README.md)


# CI/CD Integration

## GitHub Actions Workflows

CI/CD uses **trigger-based** workflows (manual run from the Actions tab). The C++ job builds and tests the native library with conda-forge dependencies (OpenVDB, ITK, libmongocxx, pybind11).

### Workflow Files

| Workflow | File | Purpose |
|----------|------|---------|
| **CI** | `.github/workflows/ci.yml` | Full test suite (Python + C++), lint, test matrix, optional performance, notebooks |
| **PR Checks** | `.github/workflows/pr.yml` | Quick validation, format check |
| **Nightly** | `.github/workflows/nightly.yml` | Weekly full suite, security scans, notebook validation |
| **Release** | `.github/workflows/release.yml` | Release validation: Python tests, C++ build/test, docs, notebooks |

### CI Jobs (ci.yml)

- **Test**: Python 3.11, `pip install -r requirements.txt`, unit / integration / e2e / property_based / utils, coverage (cov-fail-under 15).
- **C++**: System deps (cmake, ninja, libtbb-dev, etc.) + pip pybind11; third-party (OpenVDB, ITK, mongo-cxx-driver) from cache; CMake configure + build + `ctest --output-on-failure`.
- **Lint**: Black, flake8, pylint, mypy.
- **Test Matrix**: Parallel runs of unit, integration, e2e, property_based, utils.
- **Performance** (optional): Regression and benchmarks when `run_performance` is true.
- **Notebooks**: Structure and dependency validation; optional execution when `execute_notebooks` is true.

### Pre-commit / Local Checks

- Run unit tests: `pytest tests/unit -m "unit" -v`
- Check formatting: `black --check src/ tests/`
- Run C++ tests (after build): `ctest --test-dir build --output-on-failure`

### Test Execution Strategy

- **On manual CI run**: Python tests + C++ build/test + lint + (optional) performance + notebooks.
- **Nightly**: Full Python suite, security scans, notebook validation.
- **On release**: Full Python suite, C++ build and test, docs placeholder, notebook validation.

## Related

- [Success Metrics](12-success-metrics.md) - CI/CD success metrics
- [Best Practices](11-best-practices.md) - CI/CD best practices
- [Running Tests](14-running-tests.md) - Commands and CI/CD pipeline notes
- [Build Tests](15-build-tests.md) - C++ build and test setup

---

**Parent**: [Test Documentation](README.md)

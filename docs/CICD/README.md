# CI/CD Documentation

**Version**: 1.0  
**Last Updated**: 2024

## Overview

The AM-QADF project uses GitHub Actions for Continuous Integration and Continuous Deployment (CI/CD). This documentation describes the CI/CD workflows, their purposes, and how to use them.

## Quick Start

### For Developers

1. **Before Pushing**:
   ```bash
   # Run tests locally
   pytest tests/unit -m "unit" -v
   pytest tests/integration -m "integration" -v
   pytest tests/e2e -m "e2e" -v
   
   # Check formatting
   black --check src/ tests/
   
   # Validate notebooks
   python -m json.tool notebooks/00_*.ipynb > /dev/null
   ```

2. **Triggering Workflows**:
   - Workflows are **trigger-based** (manual) - they don't run automatically on push
   - Go to Actions tab → Select workflow → Click "Run workflow"
   - Choose options (branch, which tests to run, etc.)
   - Review results after execution

### For Maintainers

1. **Monitor Workflows**:
   - Check Actions tab regularly
   - Review failed workflows
   - Address flaky tests

2. **Review Coverage**:
   - Monitor coverage trends
   - Ensure new code is tested
   - Maintain coverage thresholds

## Documentation Structure

```
docs/CICD/
├── README.md                 # This file - overview and quick start
├── workflows.md              # Detailed workflow documentation
├── notebook-validation.md    # Notebook validation guide
├── local-testing.md         # Local testing with act
└── troubleshooting.md        # Troubleshooting guide
```

## Workflow Overview

| Workflow | Trigger | Purpose | Duration |
|----------|---------|----------|----------|
| **CI** | Manual (workflow_dispatch) | Full test suite, linting, notebooks | ~15-20 min |
| **PR Checks** | Manual (workflow_dispatch) | Quick validation for PRs | ~5-10 min |
| **Nightly** | Scheduled (Weekly) + Manual | Comprehensive testing | ~20-30 min |
| **Release** | Release published + Manual | Release validation | ~15-20 min |

## Workflows

### 1. Continuous Integration (CI)

**File**: `.github/workflows/ci.yml`

Comprehensive CI workflow with:
- Multi-version Python testing (3.9, 3.10, 3.11)
- Code linting and formatting checks
- Test suite matrix execution
- Performance benchmarks
- Notebook validation

[Detailed Documentation](workflows.md#1-ci-workflow-ciyml)

### 2. Pull Request Checks

**File**: `.github/workflows/pr.yml`

Quick validation for pull requests:
- Quick unit tests
- Format checking
- Import validation
- Notebook quick check

[Detailed Documentation](workflows.md#2-pr-checks-workflow-pryml)

### 3. Nightly Build

**File**: `.github/workflows/nightly.yml`

Weekly comprehensive testing:
- Full test suite across Python versions
- Security scans (Bandit, Safety)
- Code quality checks
- Comprehensive notebook validation

[Detailed Documentation](workflows.md#3-nightly-build-workflow-nightlyyml)

### 4. Release

**File**: `.github/workflows/release.yml`

Release validation:
- Full test suite
- Documentation build
- Release-ready notebook validation
- Test report generation

[Detailed Documentation](workflows.md#4-release-workflow-releaseyml)

## Key Features

### Multi-Version Testing

Tests run across Python 3.9, 3.10, and 3.11 to ensure compatibility.

### Notebook Validation

All workflows include notebook validation:
- Structure validation
- Dependency checks
- Documentation validation

[Notebook Validation Guide](notebook-validation.md)

### Code Quality

Comprehensive code quality checks:
- Black formatting
- Flake8 linting
- Pylint analysis
- MyPy type checking

### Security Scanning

Nightly builds include:
- Bandit security scan
- Safety dependency check
- Security report artifacts

## Configuration

### Python Versions

- **3.9**: Minimum supported
- **3.10**: Recommended
- **3.11**: Latest stable

### System Dependencies

- `libgl1` - OpenGL libraries
- `libglib2.0-0` - GLib libraries

### Python Dependencies

Installed from `requirements.txt`:
- Core framework dependencies
- Testing dependencies
- Linting dependencies
- Notebook dependencies

## Troubleshooting

Common issues and solutions:

- [Workflow Not Triggering](troubleshooting.md#1-workflow-not-triggering)
- [Tests Failing](troubleshooting.md#2-tests-failing)
- [Import Errors](troubleshooting.md#3-import-errors)
- [Formatting Check Fails](troubleshooting.md#4-formatting-check-fails)
- [Notebook Validation Fails](troubleshooting.md#5-notebook-validation-fails)

[Complete Troubleshooting Guide](troubleshooting.md)

## Best Practices

### For Developers

1. **Test Locally First**
   ```bash
   pytest tests/ -v
   black --check src/ tests/
   ```

2. **Check Formatting**
   ```bash
   black src/ tests/
   ```

3. **Validate Notebooks**
   ```bash
   python -m json.tool notebooks/00_*.ipynb
   ```

### For Maintainers

1. **Monitor Workflows**
   - Check Actions tab regularly
   - Review failed workflows
   - Address issues promptly

2. **Review Coverage**
   - Monitor coverage trends
   - Ensure new code is tested
   - Maintain >80% coverage

3. **Security Scans**
   - Review Bandit reports
   - Address vulnerabilities
   - Update dependencies

## Related Documentation

- **[Workflows Guide](workflows.md)** - Detailed workflow documentation
- **[Notebook Validation](notebook-validation.md)** - Notebook validation guide
- **[Local Testing](local-testing.md)** - Test workflows locally with act
- **[Troubleshooting](troubleshooting.md)** - Troubleshooting guide
- **[Testing Documentation](../Tests/README.md)** - Testing guide
- **[Notebook Documentation](../Notebook/README.md)** - Notebook documentation

## Workflow Status

Add badges to README.md:

```markdown
![CI](https://github.com/kanhaiya-gupta/AM-QADF/workflows/CI/badge.svg)
![Nightly](https://github.com/kanhaiya-gupta/AM-QADF/workflows/Nightly%20Build/badge.svg)
![Coverage](https://codecov.io/gh/kanhaiya-gupta/AM-QADF/branch/main/graph/badge.svg)
```

---

**Last Updated**: 2024

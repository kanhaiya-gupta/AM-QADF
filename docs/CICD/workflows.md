# CI/CD Workflows Detailed Guide

## Overview

This document provides detailed information about each CI/CD workflow in the AM-QADF project.

## 1. CI Workflow (`ci.yml`)

### Purpose

Comprehensive continuous integration workflow that runs tests, linting, and validation on every push and can be triggered manually.

### Trigger Events

```yaml
on:
  workflow_dispatch:
    inputs:
      branch:
        description: 'Branch to run CI on'
        required: false
        default: 'main'
        type: choice
        options:
          - main
          - develop
          - all
      run_tests:
        description: 'Run test suites'
        required: false
        default: true
        type: boolean
      run_performance:
        description: 'Run performance tests'
        required: false
        default: false
        type: boolean
```

**Note**: This workflow is **trigger-based only** - it does not run automatically on push. You must manually trigger it from the Actions tab.

### Jobs

#### Job 1: Test

**Purpose**: Run tests across multiple Python versions

**Matrix Strategy**:
- Python versions: 3.9, 3.10, 3.11
- Fail-fast: false (all versions tested even if one fails)

**Steps**:
1. Checkout code
2. Set up Python with caching
3. Install system dependencies (libgl1, libglib2.0-0)
4. Install Python dependencies
5. Run unit tests
6. Run integration tests (continue on error)
7. Run all tests with coverage
8. Upload coverage to Codecov

**Test Commands**:
```bash
# Unit tests
pytest tests/unit -m "unit" -v --tb=short

# Integration tests
pytest tests/integration -m "integration" -v --tb=short

# E2E tests
pytest tests/e2e -m "e2e" -v --tb=short

# Property-based tests
pytest tests/property_based -m "property_based" -v --tb=short

# Utils tests
pytest tests/utils -v --tb=short

# Full coverage
pytest tests/ --cov=src/am_qadf --cov-report=xml --cov-report=term-missing --cov-report=html -v --cov-fail-under=80
```

#### Job 2: Lint

**Purpose**: Code quality and style checks

**Tools Used**:
- **Black**: Code formatting
- **Flake8**: Linting
- **Pylint**: Code analysis
- **MyPy**: Type checking

**Steps**:
1. Checkout code
2. Set up Python 3.11
3. Install linting dependencies
4. Check formatting with Black
5. Lint with Flake8
6. Analyze with Pylint
7. Type check with MyPy

**Linting Commands**:
```bash
# Format check
black --check --diff src/ tests/

# Flake8 (critical errors)
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Flake8 (warnings)
flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Pylint
pylint src/ --disable=all --enable=E,F --exit-zero

# MyPy
mypy src/ --ignore-missing-imports --no-strict-optional
```

#### Job 3: Test Matrix

**Purpose**: Run test suites in parallel

**Test Suites**:
- `tests/unit` - Unit tests
- `tests/integration` - Integration tests
- `tests/e2e` - End-to-end tests
- `tests/property_based` - Property-based tests
- `tests/utils` - Utility tests

**Benefits**:
- Faster execution through parallelization
- Isolated test suite failures
- Better test organization visibility

#### Job 4: Performance

**Purpose**: Performance benchmarking

**Conditions**:
- Only runs when `run_performance` input is set to `true`
- Sequential performance tests only (Spark-required tests are skipped)
- Uses pytest-benchmark for benchmarks

**Steps**:
1. Set up environment
2. Install dependencies
3. Run performance benchmarks
4. Generate benchmark JSON

**Performance Commands**:
```bash
# Performance regression tests (sequential only)
pytest tests/performance/regression -m "performance and not requires_spark" -v --tb=short

# Performance benchmarks
pytest tests/performance/benchmarks -m "benchmark" --benchmark-only --benchmark-json=benchmark.json
```

#### Job 5: Notebooks

**Purpose**: Validate interactive notebooks

**Validation Checks**:
1. **Existence Check**: Verify notebooks directory exists
2. **Structure Validation**: Validate JSON structure
3. **Metadata Check**: Ensure proper metadata
4. **Dependency Check**: Verify required imports
5. **Documentation Check**: Validate documentation exists

**Validation Script**:
```python
# Structure validation
import json
from pathlib import Path

notebooks_dir = Path('notebooks')
for nb_file in notebooks_dir.glob('*.ipynb'):
    with open(nb_file, 'r') as f:
        nb = json.load(f)
    
    # Check structure
    assert 'cells' in nb
    assert len(nb['cells']) > 0
    assert 'metadata' in nb
```

## 2. PR Checks Workflow (`pr.yml`)

### Purpose

Quick validation workflow for pull requests to catch issues early.

### Trigger Events

```yaml
on:
  workflow_dispatch:
    inputs:
      pr_number:
        description: 'Pull Request number to check'
        required: false
        type: string
      branch:
        description: 'Target branch'
        required: false
        default: 'main'
        type: choice
        options:
          - main
          - develop
      run_quick_tests:
        description: 'Run quick validation tests'
        required: false
        default: true
        type: boolean
      run_format_check:
        description: 'Run format check'
        required: false
        default: true
        type: boolean
```

**Note**: This workflow is **trigger-based only** - it does not run automatically on pull requests. You must manually trigger it from the Actions tab.

### Jobs

#### Job 1: PR Checks

**Purpose**: Quick validation for PRs

**Steps**:
1. Checkout with full history
2. Set up Python
3. Install dependencies
4. Check for test files
5. Run quick unit tests
6. Check import structure
7. Validate pytest configuration
8. Quick notebook validation

**Validation Commands**:
```bash
# Quick unit tests
pytest tests/unit -m "unit" -v --tb=line -x

# Import check
python -c "from am_qadf import *"

# Pytest config
pytest --collect-only -q
```

#### Job 2: Format Check

**Purpose**: Ensure code formatting

**Steps**:
1. Checkout code
2. Set up Python
3. Install Black
4. Check formatting
5. Comment on PR if formatting issues found

**Format Check**:
```bash
black --check src/ tests/
```

**PR Comment** (on failure):
```
⚠️ Code formatting issues detected. Please run `black src/ tests/` to format your code.
```

## 3. Nightly Build Workflow (`nightly.yml`)

### Purpose

Comprehensive weekly testing and quality checks.

### Trigger Events

```yaml
on:
  schedule:
    # Run every Sunday at 2 AM UTC
    - cron: '0 2 * * 0'
  workflow_dispatch:
```

### Jobs

#### Job 1: Nightly Tests

**Purpose**: Full test suite across Python versions

**Matrix Strategy**:
- Python versions: 3.9, 3.10, 3.11

**Steps**:
1. Set up Python
2. Install system dependencies
3. Install dependencies
4. Run full test suite (max 5 failures)
5. Run slow tests

**Test Commands**:
```bash
# Full suite
pytest tests/ -v --tb=short --maxfail=5

# Slow tests
pytest tests/ -m "slow" -v
```

#### Job 2: Code Quality

**Purpose**: Security and quality checks

**Tools Used**:
- **Bandit**: Security linting
- **Safety**: Dependency vulnerability scanning

**Steps**:
1. Set up Python
2. Install quality tools
3. Run Bandit security scan
4. Run Safety vulnerability check
5. Upload security reports

**Security Commands**:
```bash
# Bandit scan
bandit -r src/ -f json -o bandit-report.json

# Safety check
safety check --json
```

#### Job 3: Notebooks

**Purpose**: Comprehensive notebook validation

**Validation**:
- Validates all notebooks
- Checks documentation coverage
- Reports detailed results

## 4. Release Workflow (`release.yml`)

### Purpose

Release validation and documentation build.

### Trigger Events

```yaml
on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      version:
        description: 'Version tag (e.g., v1.0.0)'
        required: true
        type: string
```

### Jobs

#### Job 1: Build and Test

**Purpose**: Validate release readiness

**Steps**:
1. Set up environment
2. Install dependencies
3. Run full test suite
4. Generate HTML test report
5. Upload test report artifact

**Test Report**:
- Format: HTML (self-contained)
- Location: `test_report.html`
- Artifact: `test-report`

#### Job 2: Documentation

**Purpose**: Build documentation for release

**Steps**:
1. Set up Python
2. Install Sphinx and theme
3. Build documentation
4. Create build directory

**Note**: Currently a placeholder - can be extended with actual Sphinx build.

#### Job 3: Notebooks

**Purpose**: Validate notebooks for release

**Validation**:
- Comprehensive structure check
- Release-ready validation
- Detailed validation report

## Workflow Execution Times

| Workflow | Typical Duration | Factors |
|----------|------------------|---------|
| CI | 15-20 min | Test matrix, multiple jobs |
| PR Checks | 5-10 min | Quick validation only |
| Nightly | 20-30 min | Full suite, security scans |
| Release | 15-20 min | Full validation, docs build |

## Workflow Dependencies

```
CI Workflow
├── Test (parallel)
├── Lint (parallel)
├── Test Matrix (parallel)
├── Performance (conditional)
└── Notebooks (parallel)

PR Checks
├── PR Checks
└── Format Check

Nightly
├── Nightly Tests (matrix)
├── Code Quality
└── Notebooks

Release
├── Build and Test
├── Documentation
└── Notebooks
```

## Best Practices

### For Workflow Development

1. **Use Matrix Strategies**: Test across multiple Python versions
2. **Parallel Jobs**: Run independent jobs in parallel
3. **Conditional Execution**: Use `if` conditions for optional jobs
4. **Continue on Error**: Use `continue-on-error: true` for non-critical checks
5. **Artifacts**: Upload reports and logs as artifacts

### For Workflow Maintenance

1. **Monitor Execution Times**: Keep workflows under 30 minutes
2. **Review Failures**: Address flaky tests promptly
3. **Update Dependencies**: Keep actions and tools updated
4. **Document Changes**: Update this documentation when workflows change

## Troubleshooting Workflows

### Workflow Not Triggering

**Check**:
- Workflow file syntax (YAML)
- Trigger conditions
- Branch names
- Event types

### Tests Failing

**Check**:
- Test output logs
- Python version compatibility
- Dependency versions
- Environment variables

### Notebook Validation Failing

**Check**:
- Notebook JSON structure
- Required metadata
- Import statements
- Documentation paths

---

**Last Updated**: 2024


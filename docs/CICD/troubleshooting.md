# CI/CD Troubleshooting Guide

## Common Issues and Solutions

### 1. Workflow Not Triggering

#### Problem
Workflow doesn't run when expected.

#### Possible Causes
- Incorrect trigger configuration
- Workflow file syntax errors
- Branch name mismatch
- Event type not matching

#### Solutions

**Check Workflow Syntax**:
```bash
# Validate YAML syntax
yamllint .github/workflows/ci.yml
```

**Verify Triggers**:
```yaml
# Check trigger configuration
on:
  push:
    branches: [main, develop]  # Verify branch names
  workflow_dispatch:           # Check manual trigger
```

**Check Branch Names**:
- Ensure branch names match exactly
- Case-sensitive matching
- Use `refs/heads/main` for main branch

### 2. Tests Failing

#### Problem
Tests pass locally but fail in CI.

#### Possible Causes
- Environment differences
- Missing dependencies
- Python version mismatch
- System dependencies missing

#### Solutions

**Check Python Version**:
```bash
# Verify Python version in workflow
python --version

# Match local version
python3.11 --version
```

**Install System Dependencies**:
```yaml
# Ensure system deps are installed
- name: Install system dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y libgl1 libglib2.0-0
```

**Check Dependencies**:
```bash
# Verify requirements.txt
pip install -r requirements.txt

# Check for missing packages
pip check
```

**Reproduce Locally**:
```bash
# Use same Python version
python3.11 -m pytest tests/ -v

# Check environment
env | grep -i python
```

### 3. Import Errors

#### Problem
Import errors in CI but not locally.

#### Possible Causes
- Missing package installation
- Incorrect Python path
- Package not in requirements.txt

#### Solutions

**Install Package**:
```bash
# Install in editable mode
pip install -e .

# Or install from source
pip install -r requirements.txt
```

**Check Python Path**:
```python
# Verify imports
python -c "import sys; print(sys.path)"
python -c "from src.analyzer import XCTAnalyzer"
```

**Update Requirements**:
```bash
# Add missing package
pip freeze > requirements.txt
```

### 4. Formatting Check Fails

#### Problem
Black formatting check fails.

#### Solutions

**Format Code**:
```bash
# Format with Black
black src/ tests/

# Check formatting
black --check src/ tests/
```

**Update Workflow**:
```yaml
# Allow formatting differences
- name: Check formatting
  run: |
    black --check --diff src/ tests/
  continue-on-error: true
```

### 5. Notebook Validation Fails

#### Problem
Notebook validation fails in CI.

#### Solutions

**Validate JSON**:
```bash
# Check JSON structure
python -m json.tool notebooks/00_*.ipynb > /dev/null
```

**Check Structure**:
```python
# Validate notebook structure
import json
with open('notebooks/00_*.ipynb', 'r') as f:
    nb = json.load(f)
assert 'cells' in nb
assert len(nb['cells']) > 0
```

**Fix Notebook**:
- Ensure valid JSON
- Add missing cells
- Add required metadata

### 6. Coverage Decreases

#### Problem
Code coverage decreases after changes.

#### Solutions

**Add Tests**:
```bash
# Write tests for new code
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Check Coverage Report**:
```bash
# Generate HTML report
pytest --cov=src --cov-report=html

# Open htmlcov/index.html
```

**Maintain Coverage**:
- Aim for >80% coverage
- Test new features
- Test edge cases

### 7. Performance Tests Fail

#### Problem
Performance benchmarks fail or timeout.

#### Solutions

**Check Benchmark**:
```bash
# Run benchmarks locally
pytest tests/performance/ --benchmark-only
```

**Adjust Timeouts**:
```yaml
# Increase timeout
- name: Run benchmarks
  timeout-minutes: 30
  run: |
    pytest tests/performance/ --benchmark-only
```

**Skip Slow Tests**:
```bash
# Skip slow tests in CI
pytest tests/ -m "not slow"
```

### 8. Security Scan Failures

#### Problem
Bandit or Safety scans find issues.

#### Solutions

**Review Findings**:
```bash
# Run Bandit locally
bandit -r src/ -f json -o bandit-report.json

# Review report
cat bandit-report.json
```

**Fix Security Issues**:
- Address high-severity issues
- Update vulnerable dependencies
- Fix security warnings

**Update Dependencies**:
```bash
# Update dependencies
pip install --upgrade package-name

# Check for vulnerabilities
safety check
```

### 9. Workflow Timeout

#### Problem
Workflow exceeds time limit.

#### Solutions

**Optimize Tests**:
```bash
# Run tests in parallel
pytest tests/ -n auto

# Skip slow tests
pytest tests/ -m "not slow"
```

**Split Workflows**:
```yaml
# Split into multiple workflows
# Fast tests in one workflow
# Slow tests in another
```

**Increase Timeout**:
```yaml
# Set job timeout
jobs:
  test:
    timeout-minutes: 60
```

### 10. Artifact Upload Fails

#### Problem
Artifact upload fails.

#### Solutions

**Check Artifact Size**:
```yaml
# Artifacts have size limits
# Compress large files
- name: Compress reports
  run: |
    tar -czf reports.tar.gz reports/
```

**Verify Path**:
```yaml
# Ensure path exists
- name: Upload artifact
  uses: actions/upload-artifact@v4
  with:
    path: reports/  # Verify path exists
```

## Debugging Strategies

### 1. Enable Debug Logging

```yaml
# Enable debug logging
- name: Debug
  env:
    ACTIONS_STEP_DEBUG: true
    ACTIONS_RUNNER_DEBUG: true
  run: |
    echo "Debug information"
```

### 2. Check Workflow Logs

1. Go to Actions tab
2. Select failed workflow
3. Click on failed job
4. Review step logs
5. Look for error messages

### 3. Reproduce Locally

```bash
# Use same environment
docker run -it ubuntu:latest bash

# Install dependencies
apt-get update
apt-get install -y python3.11 python3-pip

# Reproduce steps
pip install -r requirements.txt
pytest tests/ -v
```

### 4. Test Workflow Changes

```bash
# Test workflow syntax
yamllint .github/workflows/ci.yml

# Validate with act (local GitHub Actions)
act -l
act push
```

## Getting Help

### 1. Check Documentation

- [Workflows Guide](workflows.md)
- [Notebook Validation](notebook-validation.md)
- [Testing Documentation](../Tests/README.md)

### 2. Review Logs

- Workflow execution logs
- Test output
- Error messages

### 3. Search Issues

- Check existing GitHub issues
- Search for similar problems
- Review closed issues

### 4. Ask for Help

- Open a GitHub issue
- Use the question template
- Provide detailed information

## Prevention

### Best Practices

1. **Test Locally First**:
   ```bash
   # Run tests before pushing
   pytest tests/ -v
   black --check src/ tests/
   ```

2. **Use Pre-commit Hooks**:
   ```bash
   # Install pre-commit
   pip install pre-commit
   pre-commit install
   ```

3. **Monitor Workflows**:
   - Check workflow status regularly
   - Address failures promptly
   - Review coverage trends

4. **Keep Dependencies Updated**:
   ```bash
   # Update dependencies
   pip install --upgrade package-name
   pip freeze > requirements.txt
   ```

5. **Document Changes**:
   - Document workflow changes
   - Update troubleshooting guide
   - Share solutions with team

---

**Last Updated**: 2024


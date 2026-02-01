# Local CI/CD Testing with `act`

This guide covers local testing of GitHub Actions workflows using `act` and how to trigger workflows directly on GitHub.

## Overview

This guide covers two approaches to working with GitHub Actions workflows:

1. **Local Testing with `act`**: Test workflows locally before pushing to GitHub
2. **Triggering on GitHub**: How to manually trigger workflows on GitHub

### Local Testing with `act`

`act` is a command-line tool that runs GitHub Actions workflows locally using Docker containers. This enables you to:

- Test workflows before pushing to GitHub
- Debug workflow issues locally
- Validate workflow changes quickly
- Save CI/CD minutes and resources

### Triggering Workflows on GitHub

For production runs and automated testing, workflows can be triggered directly on GitHub through the Actions interface or automatically via events (pushes, releases, schedules).

## Prerequisites

### 1. Docker

`act` requires Docker to run workflows. Install Docker:

**On Linux/WSL:**
```bash
# Install Docker
sudo apt-get update
sudo apt-get install docker.io
sudo service docker start

# Add user to docker group (if needed)
sudo usermod -aG docker $USER
newgrp docker
```

**On Windows:**
- Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Ensure WSL2 integration is enabled if using WSL

**On macOS:**
- Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Or use Homebrew: `brew install --cask docker`

Verify Docker is running:

```bash
docker --version
docker ps
```

### 2. Install `act`

**On Linux/WSL:**
```bash
# Install act using the official installer (RECOMMENDED)
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Add to PATH (if installed in ./bin)
export PATH="$PATH:$(pwd)/bin"
echo 'export PATH="$PATH:'$(pwd)'/bin"' >> ~/.bashrc

# Verify installation
act --version

# Note: Avoid using snap version (sudo snap install act) as it may crash
# If you installed via snap and it crashes, remove it first:
# sudo snap remove act
# Then use the official installer above
```

**On macOS:**
```bash
# Using Homebrew
brew install act

# Or using the installer
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

**On Windows (WSL):**
```bash
# Use Linux installation method in WSL
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

### 3. Add to PATH (if needed)

If `act` was installed in `./bin` directory, add it to your PATH:

```bash
# Add to PATH for current session
export PATH="$PATH:$(pwd)/bin"

# Add to PATH permanently (adds to ~/.bashrc)
echo 'export PATH="$PATH:'$(pwd)'/bin"' >> ~/.bashrc

# Reload shell or source bashrc
source ~/.bashrc
```

### 4. Verify Installation

```bash
act --version
```

You should see output like: `act version 0.2.84`

## Basic Usage

### List All Workflows

See all available workflows and their events:

```bash
# From AM-QADF directory
cd AM-QADF
act -l
```

This shows:
- Workflow names
- Job names
- Available events (workflow_dispatch, schedule, etc.)

### Dry-Run Mode

Test workflows without actually running them (recommended first step):

```bash
# Test CI workflow
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -n

# Test PR workflow
act workflow_dispatch -W .github/workflows/pr.yml --input pr_number=1 --input branch=main -n

# Test nightly workflow (simulate schedule)
act schedule -W .github/workflows/nightly.yml -n

# Test release workflow
act workflow_dispatch -W .github/workflows/release.yml --input version=v1.0.0 -n
```

The `-n` flag shows what would happen without executing.

### Run Workflows

Actually execute workflows (takes longer, uses Docker):

```bash
# Run C++ only
act workflow_dispatch -W .github/workflows/ci.yml --input cpp_only=true -j cpp

# Run CI workflow
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main

# Run PR workflow
act workflow_dispatch -W .github/workflows/pr.yml --input pr_number=1 --input branch=main

# Run nightly workflow
act schedule -W .github/workflows/nightly.yml

# Run release workflow
act workflow_dispatch -W .github/workflows/release.yml --input version=v1.0.0
```

## Testing Specific Workflows

### CI Workflow

The CI workflow runs tests, linting, code quality checks, and notebook validation:

```bash
# Dry-run
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -n

# Full run
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main

# Test specific job
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j test
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j cpp
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j lint
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j notebooks
```

**Inputs:**
- `branch`: Choose branch (main, develop, or all)
- `run_tests`: Run test suites (default: true)
- `run_performance`: Run performance tests (default: false)
- `execute_notebooks`: Execute notebooks (default: false)

**Jobs:**
- `test`: Python 3.11 testing (unit, integration, e2e, property_based, utils, coverage)
- `cpp`: C++ build and test (system deps + pybind11, third-party from release, CMake, ctest)
- `lint`: Code quality checks
- `test-matrix`: Parallel test suite execution
- `performance`: Performance regression and benchmarks (when `run_performance` is true)
- `notebooks`: Notebook validation

**C++ job with act:** The C++ job downloads third-party dependencies (OpenVDB, ITK, mongo-cxx-driver) from the GitHub release **"Third-party dependencies"** (tag `v0.2.0` or as set in the workflow). No local install folders or `--bind` are needed — local act and GitHub CI behave the same. Example: `act workflow_dispatch -W .github/workflows/ci.yml --input cpp_only=true -j cpp`

**Building C++ natively (WSL/Linux without act):** If you run CMake/ninja directly on your machine using the same release zip, OpenVDB in the zip was built against **Boost 1.82**. Your system may have an older Boost (e.g. 1.74), which causes linker errors. See [Third-Party Release Asset: Build Environment](../Infrastructure/third-party/RELEASE_ASSET_BUILD.md#local-build-wsl-or-linux-with-the-release-zip) for how to provide Boost 1.82 or rebuild the zip with your system Boost.

### PR Workflow

The PR workflow validates pull requests:

```bash
# Dry-run
act workflow_dispatch -W .github/workflows/pr.yml --input pr_number=1 --input branch=main -n

# Full run
act workflow_dispatch -W .github/workflows/pr.yml --input pr_number=1 --input branch=main
```

**Inputs:**
- `pr_number`: Pull request number (optional; for PR comments)
- `branch`: Target branch (main or develop)

**Jobs:**
- `pr-checks`: Quick validation tests
- `format-check`: Code formatting validation

### Nightly Workflow

The nightly workflow runs weekly tests and security checks:

```bash
# Simulate schedule trigger
act schedule -W .github/workflows/nightly.yml -n

# Or trigger manually
act workflow_dispatch -W .github/workflows/nightly.yml -n
```

**Jobs:**
- `nightly-tests`: Full test suite (Python 3.11)
- `code-quality`: Security scans (Bandit, Safety)
- `notebooks`: Comprehensive notebook validation

### Release Workflow

The release workflow builds and tests releases:

```bash
# Quick test (dry-run)
act workflow_dispatch -W .github/workflows/release.yml --input version=v1.0.0 -n

# Or full run
act workflow_dispatch -W .github/workflows/release.yml --input version=v1.0.0

# Simulate release event (alternative)
act release -W .github/workflows/release.yml -n
```

**Inputs:**
- `version`: Version tag (e.g., v1.0.0) - **required**

**Note**: The documentation build job only runs on actual `release` events, not `workflow_dispatch`. When testing manually, only the build-and-test and notebooks jobs will run.

**Jobs:**
- `build-and-test`: Full Python test suite, C++ build and test, report generation
- `documentation`: Documentation build (only on release events)
- `notebooks`: Release-ready notebook validation

## Advanced Usage

### Test Specific Jobs

Run only specific jobs from a workflow:

```bash
# Run only the lint job
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j lint

# Run only the Python test job
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j test

# Run only the C++ build and test job
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j cpp

# Run only notebooks validation
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j notebooks

# Run multiple jobs
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j lint -j notebooks
```

### Use Different Docker Images

Specify a different runner image:

```bash
# Use medium image (default, recommended)
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -P ubuntu-latest=catthehacker/ubuntu:act-latest

# Use large image (more compatible, but ~17GB)
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -P ubuntu-latest=catthehacker/ubuntu:act-22.04
```

### Set Environment Variables

Pass environment variables to workflows:

```bash
# From .env file
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -e .env

# Or set inline
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main --env MY_VAR=value
```

### Verbose Output

Get more detailed output:

```bash
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -v
```

### List Available Events

See what events a workflow supports:

```bash
act -l -W .github/workflows/ci.yml
```

## Configuration

### Act Configuration File

Create `~/.config/act/actrc` to set defaults:

```bash
mkdir -p ~/.config/act
cat > ~/.config/act/actrc << EOF
-P ubuntu-latest=catthehacker/ubuntu:act-latest
--container-architecture linux/amd64
EOF
```

### Docker Image Selection

When first running `act`, you'll be prompted to choose a default image:

- **Large size image**: ~17GB download, most compatible, includes snapshots of GitHub Hosted Runners
- **Medium size image**: ~500MB, includes necessary tools, compatible with most actions (recommended)
- **Micro size image**: <200MB, only NodeJS, doesn't work with all actions

**Recommendation**: Choose "Medium" for most use cases.

## Common Issues and Troubleshooting

### Docker Not Running

**Error**: `Cannot connect to the Docker daemon`

**Solution**:
```bash
# Start Docker service (Linux/WSL)
sudo service docker start

# Or if using Docker Desktop, ensure it's running
# Check Docker Desktop status
```

### Permission Denied

**Error**: `permission denied while trying to connect to the Docker daemon socket`

**Solution**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or:
newgrp docker
```

### Package Installation Errors

**Error**: `Package 'libgl1-mesa-glx' has no installation candidate`

This occurs because `act` uses Ubuntu 24.04, which doesn't have the older `libgl1-mesa-glx` package.

**Solution**: The workflows have been updated to use `libgl1` instead, which is compatible with Ubuntu 24.04. If you encounter similar package errors:

1. Check the Ubuntu version in the error message
2. Update package names to match the Ubuntu version
3. Common replacements:
   - `libgl1-mesa-glx` → `libgl1` (Ubuntu 24.04+)
   - `libglib2.0-0` → `libglib2.0-0` (still works, but may show note about `libglib2.0-0t64`)

### Workflow Fails Locally but Works on GitHub

Some GitHub Actions features don't work locally:

- **GitHub API calls**: Actions that interact with GitHub API won't work
- **Secrets**: Need to be set manually or via environment variables
- **PR comments**: Won't post comments to actual PRs
- **Artifacts**: May behave differently
- **Package versions**: Local Docker images may use different Ubuntu versions than GitHub runners
- **Codecov upload**: Requires authentication tokens not available locally

**Workaround**: Use `-n` (dry-run) to validate structure, then test on GitHub for full functionality.

### Slow First Run

The first run downloads Docker images (~500MB for medium image). Subsequent runs are faster.

**Solution**: Images are cached, so only the first run is slow.

### Matrix Jobs Not Running

Matrix jobs may not all run in dry-run mode. This is normal.

**Solution**: Run without `-n` to see all matrix combinations execute.

### Codecov Upload Errors

**Error**: `Rate limit reached. Please upload with the Codecov repository upload token`

This is **expected** when running locally. Codecov requires authentication tokens that aren't available in local runs.

**Solution**: This is harmless - the workflow has `fail_ci_if_error: false` for Codecov, so it won't fail the build. Coverage reports are still generated locally in `htmlcov/` and `coverage.xml`.

### Artifact Upload Failures

**Error**: `Job 'Code Quality Check' failed` or artifact upload errors like:
- `Unable to get ACTIONS_RUNTIME_TOKEN env variable`
- `Error uploading artifact`

This occurs when workflows try to upload artifacts using `actions/upload-artifact@v3` or similar actions.

**Why it fails**:
- GitHub Actions artifact API isn't available locally
- `ACTIONS_RUNTIME_TOKEN` is a GitHub-specific environment variable not available locally
- Artifact upload actions don't work the same way in `act`
- Files may not exist if previous steps had `continue-on-error: true`

**Solution**: This is **expected and harmless** when running locally. The security checks (bandit, safety) still run and generate reports locally, but the upload step fails. On GitHub, artifact uploads work correctly.

**Note**: Even if the job shows as "failed" due to artifact upload errors, if your tests passed (e.g., "157 passed"), your code is working correctly. The failure is only in the artifact upload step, not in your actual tests or code.

**Workaround**: If you need the reports locally, check the generated files (e.g., `bandit-report.json`) in your working directory after the workflow runs.

### Notebook Validation Errors

**Error**: Notebook validation fails locally but works on GitHub

**Possible Causes**:
- Notebooks directory not found
- JSON structure issues
- Missing dependencies

**Solution**:
```bash
# Validate notebook JSON structure
python -m json.tool notebooks/00_Introduction_to_AM_QADF.ipynb > /dev/null

# Check if notebooks directory exists
ls -la notebooks/

# Verify notebook structure
python -c "
import json
from pathlib import Path
nb = json.load(open('notebooks/00_Introduction_to_AM_QADF.ipynb'))
assert 'cells' in nb
assert len(nb['cells']) > 0
"
```

### Final Cleanup Errors

**Error**: `Error occurred running finally: exitcode '1': failure`

This can occur when running workflows locally, often from:
- Conditional jobs that don't match local event context
- Cleanup steps that expect GitHub-specific environment
- Jobs with conditions like `if: github.event_name == 'push'`
- Artifact upload steps (see above)

**Solution**: This is usually harmless if your tests passed. The important jobs (tests, linting, notebooks) should complete successfully. The error is typically from cleanup/finally blocks that don't affect test results.

## Best Practices

### 1. Always Dry-Run First

```bash
# Always use -n first to validate
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -n
```

### 2. Test Workflow Changes Locally

Before pushing workflow changes:

```bash
# Test the modified workflow
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -n
```

### 3. Test Specific Jobs During Development

When developing, test only relevant jobs:

```bash
# Test only linting while fixing code style
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j lint

# Test only notebooks validation
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j notebooks
```

### 4. Use Verbose Mode for Debugging

When workflows fail, use verbose mode:

```bash
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -v
```

### 5. Validate Workflow Syntax

Use dry-run to catch syntax errors:

```bash
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -n
```

## Workflow-Specific Notes

### CI Workflow

- Tests Python 3.11 and runs C++ build/test (system deps + pybind11, third-party from release, CMake, ctest)
- Runs test matrix across unit, integration, e2e, property_based, utils
- Includes linting and code quality checks
- Performance tests run only when `run_performance=true` (default: false)
- Includes comprehensive notebook validation

### PR Workflow

- PR number input is optional (for PR comments)
- Comments won't be posted to actual PRs
- Use for validating workflow structure
- Includes quick notebook validation

### Nightly Workflow

- Can simulate schedule trigger
- Includes security checks (bandit, safety)
- Generates security reports
- Comprehensive notebook validation

### Release Workflow

- Requires version input for manual trigger
- Runs Python tests, then C++ build and test, then documentation and notebooks
- Documentation build job only runs on `release` events (not `workflow_dispatch`)
- Artifact upload may fail locally (expected)
- Release-ready notebook validation

## Quick Reference

```bash
# List all workflows
act -l

# Run only C++ build and test
act workflow_dispatch -W .github/workflows/ci.yml --input cpp_only=true -j cpp

# Dry-run CI workflow
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main --input run_tests=true --input run_performance=false -n

# Run CI workflow (with tests, no performance)
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main --input run_tests=true --input run_performance=false

# Run CI workflow (with performance tests)
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main --input run_tests=true --input run_performance=true

# Test specific job
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j test
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j cpp
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j lint
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j notebooks

# Verbose output
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -v

# Test PR workflow
act workflow_dispatch -W .github/workflows/pr.yml --input pr_number=1 --input branch=main -n

# Test nightly workflow
act schedule -W .github/workflows/nightly.yml -n

# Test release workflow (dry-run)
act workflow_dispatch -W .github/workflows/release.yml --input version=v1.0.0 -n

# Test release workflow (full run)
act workflow_dispatch -W .github/workflows/release.yml --input version=v1.0.0
```

## Triggering Workflows on GitHub

While `act` is great for local testing, you'll also need to trigger workflows directly on GitHub. Here's how to manually trigger each workflow:

### 1. CI Workflow (Tests, Linting, Code Quality, Notebooks)

1. Navigate to your repository's Actions page:
   ```
   https://github.com/YOUR_USERNAME/AM-QADF/actions
   ```
2. Click **"CI"** in the left sidebar
3. Click **"Run workflow"** button (top right)
4. Configure inputs:
   - **Branch**: `main` (or `develop`, or `all`)
   - **Run test suites**: `true` (default) - runs unit, integration, e2e, property-based, and utils tests
   - **Run performance tests**: `false` (default) - set to `true` to run performance regression tests
5. Click **"Run workflow"**

**What it does**: Runs the full Python test suite (3.11), C++ build and test, linting, code quality checks, and notebook validation. Performance tests only run if `run_performance` is set to true.

### 2. PR Checks Workflow

1. Navigate to: `https://github.com/YOUR_USERNAME/AM-QADF/actions`
2. Click **"PR Checks"** in the left sidebar
3. Click **"Run workflow"**
4. Configure inputs:
   - **PR number**: e.g., `1` (optional - for PR comments)
   - **Branch**: `main` (or `develop`)
   - **Run quick tests**: `true` (default) - runs quick validation tests
   - **Run format check**: `true` (default) - checks code formatting
5. Click **"Run workflow"**

**Note**: PR number is optional. The workflow can run without it for general validation.

**What it does**: Validates code, runs quick unit tests, checks code formatting, and performs quick notebook validation.

### 3. Release Workflow

1. Navigate to: `https://github.com/YOUR_USERNAME/AM-QADF/actions`
2. Click **"Release"** in the left sidebar
3. Click **"Run workflow"**
4. Enter **version**: e.g., `v1.0.0`
5. Click **"Run workflow"`

**What it does**: Runs full Python test suite, C++ build and test, generates test reports, and validates notebooks for a release.

**Note**: This workflow also runs automatically when you publish a GitHub release. When manually triggered, it runs the full test suite with coverage reporting.

### 4. Nightly Workflow

The nightly workflow runs automatically **weekly** (Sundays at 2 AM UTC). You can also trigger it manually:

1. Navigate to: `https://github.com/YOUR_USERNAME/AM-QADF/actions`
2. Click **"Nightly Build"** in the left sidebar
3. Click **"Run workflow"**
4. Click **"Run workflow"** (no inputs needed)

**What it does**: Runs comprehensive test suite (Python 3.11), performs security checks, and validates notebooks.

### Workflow Status

After triggering a workflow, you can:
- **View progress**: Click on the workflow run to see real-time logs
- **Check results**: Green checkmark = success, red X = failure
- **Download artifacts**: Test reports, coverage reports, security reports, etc. (if generated)
- **Re-run failed jobs**: Click "Re-run jobs" if something fails

### GitHub Actions Settings

Before workflows can run, ensure:

1. **Actions are enabled**: 
   - Go to: `Settings` → `Actions` → `General`
   - Under "Actions permissions", select "Allow all actions and reusable workflows"

2. **Workflow permissions**:
   - Default permissions (read/write) are usually sufficient
   - Can be adjusted in `Settings` → `Actions` → `General` → `Workflow permissions`

## Testing Notebooks Locally

You can test Jupyter notebooks locally before running them in CI/CD workflows.

### Using the Test Script

A Python script is provided to test notebooks:

```bash
# Test all notebooks in the notebooks/ directory
python scripts/test_notebooks.py

# Test a specific notebook
python scripts/test_notebooks.py notebooks/00_Introduction.ipynb

# Test notebooks in a specific directory
python scripts/test_notebooks.py notebooks/

# Custom timeout (default: 300 seconds)
python scripts/test_notebooks.py --timeout 600

# Custom skip patterns
python scripts/test_notebooks.py --skip "_not_good" "demo/"
```

### Manual Testing with nbconvert

You can also test notebooks manually using `jupyter nbconvert`:

```bash
# Install nbconvert if not already installed
pip install jupyter nbconvert

# Execute a single notebook
jupyter nbconvert --to notebook --execute notebooks/00_Introduction.ipynb

# Execute with timeout (5 minutes)
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=300 \
  --ExecutePreprocessor.kernel_name=python3 \
  notebooks/00_Introduction.ipynb
```

### Testing in CI/CD Workflows

Notebook execution is available in all workflows:

1. **CI Workflow**: Enable with `execute_notebooks=true` input
   ```bash
   act workflow_dispatch \
     -W .github/workflows/ci.yml \
     --input execute_notebooks=true \
     --job notebooks
   ```

2. **PR Checks**: Enable with `execute_notebooks=true` input
   ```bash
   act workflow_dispatch \
     -W .github/workflows/pr.yml \
     --input execute_notebooks=true \
     --job pr-checks
   ```

3. **Release Workflow**: Notebooks are automatically executed for releases
4. **Nightly Build**: Notebooks are automatically executed (subset of 10)

### Notebook Execution Features

- **Automatic skipping**: Notebooks matching patterns like `_not_good`, `_old`, `demo/`, `Demo`, `test_` are skipped (case-insensitive)
- **Timeout protection**: Each notebook has a timeout (default: 5 minutes)
- **Error reporting**: Detailed error messages for failed notebooks
- **CI environment**: Automatically sets `NUMBA_DISABLE_JIT=1` and `CI=true` to prevent crashes

### Troubleshooting Notebook Execution

**Issue**: `jupyter nbconvert not found`
- **Solution**: Install with `pip install jupyter nbconvert`

**Issue**: Notebook times out
- **Solution**: Increase timeout with `--timeout` flag or check for infinite loops

**Issue**: Import errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Segmentation faults in CI
- **Solution**: Already handled with `NUMBA_DISABLE_JIT=1` environment variable

## Additional Resources

- [act Documentation](https://github.com/nektos/act)
- [act Usage Guide](https://nektosact.com/usage/index.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)
- [AM-QADF CI/CD Documentation](README.md)
- [Jupyter nbconvert Documentation](https://nbconvert.readthedocs.io/)

## Summary

Using `act` for local CI/CD testing provides:

✅ **Faster feedback** - Test workflows without pushing to GitHub  
✅ **Cost savings** - Save CI/CD minutes  
✅ **Better debugging** - Debug issues locally with verbose output  
✅ **Validation** - Catch workflow errors before they reach GitHub  
✅ **Development efficiency** - Test workflow changes quickly  
✅ **Notebook validation** - Test notebook validation locally before pushing  

Remember: Always use `-n` (dry-run) first, then run actual tests when needed. Some GitHub-specific features won't work locally, but workflow structure and most steps can be validated.

---

**Last Updated**: 2025


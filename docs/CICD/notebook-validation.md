# Notebook Validation in CI/CD

## Overview

All CI/CD workflows include notebook validation to ensure the 23 interactive Jupyter notebooks maintain quality and consistency.

## Validation Checks

### 1. Structure Validation

Validates that notebooks have proper JSON structure:

```python
# Required structure
{
    "cells": [...],      # Must exist and be non-empty
    "metadata": {...},   # Must exist
    "nbformat": 4,      # Jupyter format version
    "nbformat_minor": 4
}
```

**Checks**:
- ✅ Valid JSON syntax
- ✅ Has `cells` array
- ✅ Cells array is not empty
- ✅ Has `metadata` object
- ✅ Proper nbformat version

### 2. Dependency Validation

Checks for required dependencies in notebook code:

**Required Imports**:
- `ipywidgets` - For interactive widgets
- `matplotlib` - For visualizations
- `numpy` - For numerical operations
- `pandas` - For data manipulation

**Validation**:
```python
# Checks for import statements in code cells
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        # Verify required imports are present
```

### 3. Documentation Validation

Ensures notebook documentation exists:

**Documentation Structure**:
```
docs/Notebook/
├── 04-notebooks/
│   ├── 00-introduction.md
│   ├── 01-data-query.md
│   └── ...
```

**Checks**:
- ✅ Documentation directory exists
- ✅ Key notebooks have documentation
- ✅ Documentation files are valid Markdown

### 4. Release Validation

Additional checks for release-ready notebooks:

**Release Checks**:
- ✅ All notebooks pass structure validation
- ✅ All notebooks have documentation
- ✅ No critical errors in notebooks
- ✅ Dependencies are properly declared

## Validation in Each Workflow

### CI Workflow

**Job**: `notebooks`

**Checks**:
1. Notebook files exist
2. Structure validation
3. Dependency checks
4. Documentation validation

**Output**:
```
Found 23 notebook(s)
✅ All notebooks have valid structure
✅ Notebook dependency check completed
✅ Notebook documentation directory found
```

### PR Checks Workflow

**Job**: `pr-checks` (quick validation)

**Checks**:
- Quick structure check on first 3 notebooks
- Basic validation only

**Output**:
```
Found 23 notebook(s) in PR
✅ 00_Introduction_to_AM_QADF.ipynb: Valid structure
✅ 01_Data_Query_and_Access.ipynb: Valid structure
✅ 02_Voxel_Grid_Creation.ipynb: Valid structure
```

### Nightly Workflow

**Job**: `notebooks` (comprehensive)

**Checks**:
1. Validate all notebooks
2. Check documentation coverage
3. Detailed validation report

**Output**:
```
📊 Notebook Validation Results:
✅ Valid: 23
❌ Invalid: 0

✅ Valid notebooks:
  - 00_Introduction_to_AM_QADF.ipynb
  - 01_Data_Query_and_Access.ipynb
  ...

📚 Checking notebook documentation...
✅ Key notebooks have documentation
```

### Release Workflow

**Job**: `notebooks` (release validation)

**Checks**:
1. Release-ready validation
2. Comprehensive structure check
3. Validation report for release notes

**Output**:
```
📊 Release Notebook Validation:
✅ 00_Introduction_to_AM_QADF.ipynb
✅ 01_Data_Query_and_Access.ipynb
...

✅ 23/23 notebooks passed validation
```

## Validation Scripts

### Structure Validation

```python
import json
from pathlib import Path

def validate_notebook_structure(nb_file):
    """Validate notebook JSON structure."""
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    errors = []
    
    # Check cells
    if 'cells' not in nb:
        errors.append('Missing cells')
    elif len(nb['cells']) == 0:
        errors.append('Empty cells array')
    
    # Check metadata
    if 'metadata' not in nb:
        errors.append('Missing metadata')
    
    return errors
```

### Dependency Check

```python
def check_notebook_dependencies(nb_file):
    """Check for required dependencies."""
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    required_imports = ['ipywidgets', 'matplotlib', 'numpy']
    found_imports = []
    
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            for imp in required_imports:
                if imp in source.lower() and 'import' in source:
                    found_imports.append(imp)
    
    missing = set(required_imports) - set(found_imports)
    return missing
```

### Documentation Check

```python
def check_notebook_documentation():
    """Check if notebooks have documentation."""
    from pathlib import Path
    
    doc_dir = Path('docs/Notebook/04-notebooks')
    notebooks_dir = Path('notebooks')
    
    if not notebooks_dir.exists():
        return []
    
    nb_files = {f.stem for f in notebooks_dir.glob('*.ipynb')}
    doc_files = {f.stem.replace('-', '_') for f in doc_dir.glob('*.md')}
    
    missing = []
    for nb_name in nb_files:
        # Convert notebook name to doc name
        doc_name = nb_name.lower().replace('_', '-')
        if doc_name not in doc_files:
            missing.append(nb_name)
    
    return missing
```

## Common Validation Errors

### 1. Invalid JSON

**Error**: `json.JSONDecodeError`

**Cause**: Malformed JSON in notebook file

**Fix**:
```bash
# Validate JSON
python -m json.tool notebook.ipynb > /dev/null
```

### 2. Missing Cells

**Error**: `Missing cells` or `Empty cells array`

**Cause**: Notebook has no cells or cells array is empty

**Fix**: Add at least one cell to the notebook

### 3. Missing Metadata

**Error**: `Missing metadata`

**Cause**: Notebook missing metadata section

**Fix**: Add metadata section:
```json
{
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    }
  }
}
```

### 4. Missing Documentation

**Error**: `Missing documentation for: notebook_name`

**Cause**: Notebook doesn't have corresponding documentation

**Fix**: Create documentation file in `docs/Notebook/04-notebooks/`

## Best Practices

### For Notebook Developers

1. **Validate Before Committing**:
   ```bash
   # Check JSON structure
   python -m json.tool notebooks/00_*.ipynb > /dev/null
   ```

2. **Test Locally**:
   ```bash
   # Run notebook validation script
   python scripts/validate_notebooks.py
   ```

3. **Check Dependencies**:
   - Ensure all required imports are present
   - Test notebook in clean environment

4. **Update Documentation**:
   - Create/update documentation when adding notebooks
   - Follow documentation naming conventions

### For CI/CD Maintenance

1. **Monitor Validation Results**:
   - Check workflow logs regularly
   - Address validation failures promptly

2. **Update Validation Rules**:
   - Add new checks as needed
   - Keep validation scripts up to date

3. **Document Changes**:
   - Update this documentation when validation changes
   - Document new validation rules

## Validation Configuration

### Notebook Requirements

- **Format**: Jupyter Notebook format 4.x
- **Structure**: Must have cells and metadata
- **Dependencies**: Must import required libraries
- **Documentation**: Must have corresponding doc file

### Validation Thresholds

- **Structure**: 100% must pass
- **Dependencies**: All required imports must be present
- **Documentation**: Key notebooks (00, 01, 02, 17) must have docs

## Troubleshooting

### Validation Fails in CI

1. **Check Workflow Logs**:
   - Go to Actions → Failed workflow → View logs
   - Look for validation error messages

2. **Reproduce Locally**:
   ```bash
   # Run validation script
   python -c "
   import json
   from pathlib import Path
   # ... validation code ...
   "
   ```

3. **Fix Issues**:
   - Fix JSON structure
   - Add missing dependencies
   - Create missing documentation

### False Positives

If validation incorrectly fails:

1. **Check Validation Logic**:
   - Review validation script
   - Verify notebook structure

2. **Update Validation**:
   - Adjust validation rules if needed
   - Update validation scripts

3. **Document Exception**:
   - Document why exception is needed
   - Update validation documentation

---

**Last Updated**: 2024


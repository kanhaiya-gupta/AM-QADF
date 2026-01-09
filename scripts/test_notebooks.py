#!/usr/bin/env python3
"""
Script to test Jupyter notebooks by executing them.

Usage:
    python scripts/test_notebooks.py [notebook_path]
    python scripts/test_notebooks.py notebooks/  # Test all notebooks in directory
    python scripts/test_notebooks.py notebooks/00_Introduction.ipynb  # Test single notebook
"""

import json
import subprocess
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Dict


def validate_notebook_structure(notebook_path: Path) -> Tuple[bool, str]:
    """
    Validate that a notebook has valid JSON structure and required fields.
    
    Args:
        notebook_path: Path to the notebook file
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Check basic structure
        if 'cells' not in nb:
            return False, f'{notebook_path}: Missing cells'
        elif len(nb['cells']) == 0:
            return False, f'{notebook_path}: Empty notebook'
        
        # Check for metadata
        if 'metadata' not in nb:
            return False, f'{notebook_path}: Missing metadata'
        
        return True, ""
        
    except json.JSONDecodeError as e:
        return False, f'{notebook_path}: Invalid JSON - {e}'
    except Exception as e:
        return False, f'{notebook_path}: Error - {e}'


def validate_all_notebooks_structure(notebook_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate structure of all notebooks in a directory.
    
    Args:
        notebook_dir: Directory containing notebooks
        
    Returns:
        Tuple of (all_valid: bool, error_messages: List[str])
    """
    errors = []
    
    for nb_file in notebook_dir.glob('*.ipynb'):
        # Skip .nbconvert files
        if '.nbconvert' in nb_file.name:
            continue
        
        is_valid, error = validate_notebook_structure(nb_file)
        if not is_valid:
            errors.append(error)
    
    return len(errors) == 0, errors


def check_notebook_dependencies(notebook_dir: Path) -> bool:
    """
    Basic check for notebook dependencies (informational only).
    
    Args:
        notebook_dir: Directory containing notebooks
        
    Returns:
        True if check completed (no errors)
    """
    for nb_file in notebook_dir.glob('*.ipynb'):
        # Skip .nbconvert files
        if '.nbconvert' in nb_file.name:
            continue
        
        try:
            with open(nb_file, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            # Check for import statements in code cells
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    # Basic check - could be improved
                    if 'ipywidgets' in source.lower() and 'import' in source:
                        break
        except Exception as e:
            print(f'Warning: Could not check {nb_file}: {e}')
    
    return True


def execute_notebook(notebook_path: Path, timeout: int = 300) -> Tuple[bool, str, float]:
    """
    Execute a Jupyter notebook and return success status, error message, and execution time.
    
    Args:
        notebook_path: Path to the notebook file
        timeout: Timeout in seconds per notebook
        
    Returns:
        Tuple of (success: bool, error_message: str, elapsed_time: float)
    """
    start_time = time.time()
    
    # Create temporary directory for output files
    temp_dir = tempfile.mkdtemp(prefix='nbconvert_')
    
    try:
        result = subprocess.run(
            [
                'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                f'--ExecutePreprocessor.timeout={timeout}',
                '--ExecutePreprocessor.kernel_name=python3',
                '--ExecutePreprocessor.allow_errors=False',
                '--output-dir', temp_dir,  # Output to temp directory
                str(notebook_path)
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 60  # Add buffer for nbconvert overhead
        )
        
        elapsed = time.time() - start_time
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass  # Ignore cleanup errors
        
        if result.returncode == 0:
            return True, "", elapsed
        else:
            error_msg = result.stderr[:500] if result.stderr else result.stdout[:500]
            return False, error_msg, elapsed
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
        return False, f"Timeout after {timeout}s", elapsed
    except FileNotFoundError:
        return False, "jupyter nbconvert not found. Install with: pip install jupyter nbconvert", 0
    except Exception as e:
        elapsed = time.time() - start_time
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
        return False, str(e), elapsed


def test_notebooks(notebook_paths: List[Path], skip_patterns: List[str] = None, limit: int = None) -> Dict:
    """
    Test multiple notebooks.
    
    Args:
        notebook_paths: List of notebook file paths
        skip_patterns: List of patterns to skip (e.g., ['_not_good', 'demo/'])
        limit: Maximum number of notebooks to test (None for all)
        
    Returns:
        Dictionary with 'passed', 'failed', 'skipped' lists
    """
    if skip_patterns is None:
        skip_patterns = ['_not_good', '_old', 'demo/', 'Demo', 'test_', '_test', '.nbconvert']
    
    results = {'passed': [], 'failed': [], 'skipped': []}
    tested_count = 0
    
    for nb_file in sorted(notebook_paths):
        # Apply limit if specified
        if limit is not None and tested_count >= limit:
            break
        # Skip notebooks matching skip patterns (case-insensitive for filename)
        should_skip = False
        for pattern in skip_patterns:
            # Check both full path and filename (case-insensitive)
            if pattern.lower() in str(nb_file).lower() or pattern.lower() in nb_file.name.lower():
                should_skip = True
                break
        
        if should_skip:
            results['skipped'].append(nb_file.name)
            print(f'⏭️  Skipping {nb_file.name} (matches skip pattern)')
            continue
        
        print(f'\n📓 Executing {nb_file.name}...')
        
        success, error, elapsed = execute_notebook(nb_file)
        tested_count += 1
        
        if success:
            results['passed'].append(nb_file.name)
            print(f'✅ {nb_file.name} executed successfully ({elapsed:.1f}s)')
        else:
            results['failed'].append((nb_file.name, error))
            print(f'❌ {nb_file.name} failed: {error[:200]}')
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test Jupyter notebooks by executing them',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='notebooks',
        help='Path to notebook file or directory (default: notebooks/)'
    )
    parser.add_argument(
        '--mode',
        choices=['validate', 'check-deps', 'execute'],
        default='execute',
        help='Mode: validate (structure), check-deps (dependencies), or execute (default: execute)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout per notebook in seconds (default: 300, only for execute mode)'
    )
    parser.add_argument(
        '--skip',
        nargs='+',
        default=['_not_good', '_old', 'demo/', 'Demo', 'test_', '_test', '.nbconvert'],
        help='Patterns to skip (default: _not_good _old demo/ Demo test_ _test .nbconvert)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of notebooks to test (for PR checks, use --limit 5)'
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    # Collect notebook files (exclude .nbconvert files)
    if path.is_file() and path.suffix == '.ipynb':
        if '.nbconvert' in path.name:
            print(f'⏭️  Skipping {path.name} (nbconvert artifact)')
            sys.exit(0)
        notebook_paths = [path]
        notebook_dir = path.parent
    elif path.is_dir():
        notebook_paths = [p for p in path.glob('*.ipynb') if '.nbconvert' not in p.name]
        notebook_dir = path
        if not notebook_paths and args.mode != 'check-deps':
            print(f'❌ No notebooks found in {path}')
            sys.exit(1)
    else:
        print(f'❌ Invalid path: {path}')
        sys.exit(1)
    
    # Handle different modes
    if args.mode == 'validate':
        print(f'🔍 Validating notebook structure...\n')
        all_valid, errors = validate_all_notebooks_structure(notebook_dir)
        
        if errors:
            print('❌ Notebook validation errors:')
            for error in errors:
                print(f'  - {error}')
            sys.exit(1)
        else:
            print('✅ All notebooks have valid structure')
            sys.exit(0)
    
    elif args.mode == 'check-deps':
        print(f'🔍 Checking notebook dependencies...\n')
        check_notebook_dependencies(notebook_dir)
        print('✅ Notebook dependency check completed')
        sys.exit(0)
    
    elif args.mode == 'execute':
        if args.limit:
            print(f'🚀 Testing up to {args.limit} notebook(s) from {len(notebook_paths)} total...\n')
        else:
            print(f'🚀 Testing {len(notebook_paths)} notebook(s)...\n')
        
        # Test notebooks
        results = test_notebooks(notebook_paths, skip_patterns=args.skip, limit=args.limit)
        
        # Print summary
        print(f'\n📊 Notebook Execution Results:')
        print(f'✅ Passed: {len(results["passed"])}')
        print(f'❌ Failed: {len(results["failed"])}')
        print(f'⏭️  Skipped: {len(results["skipped"])}')
        
        if results['passed']:
            print(f'\n✅ Successful notebooks:')
            for nb in results['passed']:
                print(f'   - {nb}')
        
        if results['failed']:
            print(f'\n❌ Failed notebooks:')
            for nb, error in results['failed']:
                print(f'   - {nb}: {error[:100]}')
            sys.exit(1)
        
        if results['skipped']:
            print(f'\n⏭️  Skipped notebooks:')
            for nb in results['skipped']:
                print(f'   - {nb}')
        
        print(f'\n✅ All executed notebooks passed!')


if __name__ == '__main__':
    main()

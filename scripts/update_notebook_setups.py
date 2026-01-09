"""
Script to update all notebook setup cells with standardized code.

This script updates the path setup and adds environment variable loading
to all notebooks in the notebooks/ directory.
"""

import json
import re
from pathlib import Path

# Standardized path setup code
STANDARD_PATH_SETUP = """# Add parent directory and src directory to path for imports
notebook_dir = Path().resolve()
project_root = notebook_dir.parent
src_dir = project_root / 'src'

# Add project root to path (for src.infrastructure imports)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src directory to path (for am_qadf imports)
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))"""

# Environment variable loading code
ENV_LOADING_CODE = """# Load environment variables from development.env
import os
env_file = project_root / 'development.env'
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip('"\\'')
                os.environ[key] = value
    print("✅ Environment variables loaded from development.env")
"""

def update_notebook_setup(notebook_path: Path):
    """Update setup cell in a notebook."""
    print(f"Processing {notebook_path.name}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    updated = False
    
    # Find the setup cell (usually cell 1, contains "# Setup:")
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Check if this is a setup cell
            if '# Setup:' in source or 'Setup: Import' in source:
                # Update path setup
                old_path_pattern = r'# Add parent directory to path for imports\nnotebook_dir = Path\(\)\.resolve\(\)\nif str\(notebook_dir\.parent\) not in sys\.path:\n    sys\.path\.insert\(0, str\(notebook_dir\.parent\)\)'
                
                if re.search(old_path_pattern, source):
                    # Replace old path setup
                    new_source = re.sub(
                        old_path_pattern,
                        STANDARD_PATH_SETUP,
                        source
                    )
                    
                    # Check if env loading is already present
                    if 'Load environment variables from development.env' not in new_source:
                        # Find where to insert env loading (after core imports, before module-specific imports)
                        # Look for pattern like "from typing import" or "# Try to import"
                        insert_pattern = r'(from typing import[^\n]*\n)'
                        match = re.search(insert_pattern, new_source)
                        if match:
                            insert_pos = match.end()
                            new_source = new_source[:insert_pos] + '\n' + ENV_LOADING_CODE + new_source[insert_pos:]
                        else:
                            # Try to find after numpy/pandas imports
                            insert_pattern = r'(import numpy as np\n)'
                            match = re.search(insert_pattern, new_source)
                            if match:
                                insert_pos = match.end()
                                new_source = new_source[:insert_pos] + '\n' + ENV_LOADING_CODE + new_source[insert_pos:]
                    
                    # Update imports to use am_qadf instead of old paths
                    new_source = re.sub(
                        r'from src\.data_pipeline\.data_warehouse_clients\.([^\s]+) import',
                        r'from am_qadf.\1 import',
                        new_source
                    )
                    
                    # Update exception handling to include error messages
                    new_source = re.sub(
                        r'except ImportError:\s+try:\s+from src\.data_pipeline[^\n]+\nexcept ImportError:\s+print\("⚠️[^"]+"\)',
                        r'except ImportError as e:\n    print(f"⚠️ ... not available: {e} - using demo mode")',
                        new_source,
                        flags=re.DOTALL
                    )
                    
                    # Split back into lines
                    cell['source'] = new_source.splitlines(keepends=True)
                    updated = True
                    print(f"  ✅ Updated setup cell")
                    break
    
    if updated:
        # Write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        return True
    else:
        print(f"  ⚠️  No setup cell found or already updated")
        return False

def main():
    """Update all notebooks."""
    notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    
    # Get all notebook files (excluding _not_good)
    notebook_files = sorted([
        f for f in notebooks_dir.glob('*.ipynb')
        if '_not_good' not in f.name
    ])
    
    print(f"Found {len(notebook_files)} notebooks to process\n")
    
    updated_count = 0
    for notebook_path in notebook_files:
        if update_notebook_setup(notebook_path):
            updated_count += 1
    
    print(f"\n✅ Updated {updated_count}/{len(notebook_files)} notebooks")

if __name__ == '__main__':
    main()



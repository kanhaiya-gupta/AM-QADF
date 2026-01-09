#!/usr/bin/env python
"""
AM-QADF Environment Checker

Checks if required packages for AM-QADF interactive notebooks are installed
in the amenv conda environment and helps register the kernel.
"""
import sys
import subprocess
from pathlib import Path

print("=" * 60)
print("AM-QADF Environment Checker")
print("Checking amenv kernel and notebook dependencies")
print("=" * 60)
print()

# Check current Python
print(f"Current Python: {sys.executable}")
print(f"Python version: {sys.version}")
print()

# Try to import ipywidgets
try:
    import ipywidgets as widgets
    print(f"✅ ipywidgets is installed!")
    print(f"   Version: {widgets.__version__}")
    print(f"   Location: {widgets.__file__}")
except ImportError as e:
    print(f"❌ ipywidgets is NOT installed in this environment!")
    print(f"   Error: {e}")
    print()
    print("   Install with:")
    print("   pip install ipywidgets")
    print()

# Check if ipykernel is installed
try:
    import ipykernel
    print(f"✅ ipykernel is installed!")
    print(f"   Version: {ipykernel.__version__}")
except ImportError:
    print(f"❌ ipykernel is NOT installed!")
    print("   Install with: pip install ipykernel")

print()

# Check other notebook dependencies
print("=" * 60)
print("Checking Additional Notebook Dependencies")
print("=" * 60)
print()

# Check matplotlib
try:
    import matplotlib
    print(f"✅ matplotlib is installed! (Version: {matplotlib.__version__})")
except ImportError:
    print(f"❌ matplotlib is NOT installed!")
    print("   Install with: pip install matplotlib")

# Check numpy
try:
    import numpy
    print(f"✅ numpy is installed! (Version: {numpy.__version__})")
except ImportError:
    print(f"❌ numpy is NOT installed!")
    print("   Install with: pip install numpy")

# Check pandas
try:
    import pandas
    print(f"✅ pandas is installed! (Version: {pandas.__version__})")
except ImportError:
    print(f"❌ pandas is NOT installed!")
    print("   Install with: pip install pandas")

# Check PyVista (for 3D visualization)
try:
    import pyvista
    print(f"✅ pyvista is installed! (Version: {pyvista.__version__})")
except ImportError:
    print(f"⚠️  pyvista is NOT installed (optional for 3D visualization)")
    print("   Install with: pip install pyvista")

# Check pymongo
try:
    import pymongo
    print(f"✅ pymongo is installed! (Version: {pymongo.__version__})")
except ImportError:
    print(f"❌ pymongo is NOT installed!")
    print("   Install with: pip install pymongo")

print()
print("=" * 60)
print("Kernel Registration Instructions")
print("=" * 60)
print()
print("To register amenv as a Jupyter kernel for AM-QADF notebooks:")
print()
print("1. Activate amenv:")
print("   conda activate amenv")
print()
print("2. Install ipykernel if not installed:")
print("   pip install ipykernel")
print()
print("3. Register the kernel:")
print("   python -m ipykernel install --user --name amenv --display-name 'Python (amenv - AM-QADF)'")
print()
print("4. Verify it's registered:")
print("   jupyter kernelspec list")
print()
print("5. Restart Jupyter and select 'Python (amenv - AM-QADF)' as the kernel")
print()
print("=" * 60)
print("AM-QADF Notebook Requirements")
print("=" * 60)
print()
print("For full AM-QADF notebook functionality, ensure these are installed:")
print("  - ipywidgets (for interactive widgets)")
print("  - ipykernel (for Jupyter kernel)")
print("  - matplotlib (for plotting)")
print("  - numpy (for numerical operations)")
print("  - pandas (for data manipulation)")
print("  - pymongo (for MongoDB access)")
print("  - pyvista (optional, for 3D visualization)")
print()
print("Install all with:")
print("  pip install ipywidgets ipykernel matplotlib numpy pandas pymongo pyvista")
print("=" * 60)






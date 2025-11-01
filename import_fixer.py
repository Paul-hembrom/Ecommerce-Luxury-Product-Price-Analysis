"""
UNIVERSAL IMPORT FIXER for Farfetch Pricing Analytics
Add this to EVERY Python file at the top
"""

import sys
import os
from pathlib import Path


def fix_imports():
    """Fix Python path for all imports in the project"""
    # Get the project root directory
    PROJECT_ROOT = Path(__file__).parent

    # Add these paths to Python path
    paths_to_add = [
        PROJECT_ROOT,  # Root directory
        PROJECT_ROOT / "src",  # src directory
        PROJECT_ROOT / "src" / "utils",  # utils directory
    ]

    for path in paths_to_add:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return PROJECT_ROOT


# Run this automatically
fix_imports()
print(" Import fixer activated - Python path configured")
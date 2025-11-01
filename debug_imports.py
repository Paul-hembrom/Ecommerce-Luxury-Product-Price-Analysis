"""
Debug script to identify exact import issues
"""

import sys
import os
from pathlib import Path

def debug_import(module_path):
    """Debug a specific import"""
    try:
        __import__(module_path)
        print(f" {module_path} - SUCCESS")
        return True
    except ImportError as e:
        print(f" {module_path} - FAILED: {e}")
        return False
    except Exception as e:
        print(f" {module_path} - UNEXPECTED: {e}")
        return False

# Test critical imports
print(" Debugging imports...")
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

critical_imports = [
    "utils.logger",
    "data_pipeline.process_data",
    "pricing.price_competitiveness",
    "pricing.discount_recommendation",
    "dashboards.streamlit_app"
]

all_success = True
for imp in critical_imports:
    if not debug_import(imp):
        all_success = False

if all_success:
    print("\n All critical imports working!")
else:
    print("\n Some imports failed. Check the errors above.")
"""
TEST SCRIPT - Run this first to verify all imports work
"""

import sys
import os
from pathlib import Path


def test_import(module_name, import_path):
    """Test a specific import"""
    try:
        # First, set up the path
        PROJECT_ROOT = Path(__file__).parent
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        # Try to import
        exec(f"from {import_path} import {module_name}")
        print(f" {module_name} from {import_path}")
        return True
    except ImportError as e:
        print(f" {module_name} from {import_path} - FAILED: {e}")
        return False
    except Exception as e:
        print(f" {module_name} from {import_path} - ERROR: {e}")
        return False


print(" TESTING ALL IMPORTS...")

# Test critical imports
imports_to_test = [
    ("setup_logger", "utils.logger"),
    ("DataProcessingPipeline", "data_pipeline.process_data"),
    ("PriceCompetitivenessAnalyzer", "pricing.price_competitiveness"),
    ("DiscountRecommender", "pricing.discount_recommendation"),
    ("PricingDashboard", "dashboards.streamlit_app"),
]

success_count = 0
for module_name, import_path in imports_to_test:
    if test_import(module_name, import_path):
        success_count += 1

print(f"\n RESULTS: {success_count}/{len(imports_to_test)} imports successful")

if success_count == len(imports_to_test):
    print(" ALL IMPORTS WORKING! You can run main.py")
else:
    print(" Some imports failed. Let's debug...")

    # Debug: Show current Python path
    print("\n Current Python path:")
    for path in sys.path:
        print(f"  {path}")
"""
Testing the main.py file, to confirm that everything is working properly, If this works than we can successfully execute main.py !!!
"""

import sys
import os

# NUCLEAR OPTION: Add every possible path
current_dir = os.path.dirname(os.path.abspath(__file__))
paths_to_add = [
    current_dir,  # Project root
    os.path.join(current_dir, 'src'),  # src directory
    os.path.join(current_dir, 'src', 'utils'),
    os.path.join(current_dir, 'src', 'pricing'),
    os.path.join(current_dir, 'src', 'data_pipeline'),
    os.path.join(current_dir, 'src', 'dashboards'),
    os.path.join(current_dir, 'src', 'api'),
    os.path.join(current_dir, 'src', 'forecasting'),
    os.path.join(current_dir, 'src', 'llm'),
    os.path.join(current_dir, 'src', 'reports'),
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

print("🔧 Python path configured with nuclear option")


# Now try imports with multiple fallbacks
def try_import(module_path, class_name):
    """Try multiple import strategies"""
    strategies = [
        lambda: __import__(module_path, fromlist=[class_name]),
        lambda: __import__(f"src.{module_path}", fromlist=[class_name]),
        lambda: exec(f"from {module_path} import {class_name}"),
    ]

    for i, strategy in enumerate(strategies):
        try:
            return strategy()
        except (ImportError, Exception) as e:
            if i == len(strategies) - 1:  # Last strategy failed
                print(f" All import strategies failed for {class_name}")
                raise e


try:
    # Import using our bulletproof method
    print(" Attempting imports...")

    from utils.logger import setup_logger

    print(" Logger imported")

    from data_pipeline.process_data import DataProcessingPipeline

    print(" Data pipeline imported")

    logger = setup_logger(__name__)


    def main():
        logger.info("Starting bulletproof pipeline")

        # Your data path
        data_path = "data/raw/products_350k.json"

        if not os.path.exists(data_path):
            print(f" Data file missing: {data_path}")
            return

        try:
            pipeline = DataProcessingPipeline(data_path)
            results = pipeline.run_complete_pipeline()
            print(" SUCCESS! Pipeline completed")
        except Exception as e:
            print(f" Pipeline error: {e}")


    if __name__ == "__main__":
        main()

except Exception as e:
    print(f" CRITICAL ERROR: {e}")
    print("This should not happen with nuclear option!")
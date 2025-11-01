"""
Utility to create the complete project directory structure
"""
import os
from pathlib import Path


def create_project_structure():
    """Create the complete project directory structure"""
    base_dir = Path(".")

    # Define all directories to create
    directories = [
        # Data directories
        "data/raw",
        "data/processed",
        "data/external",
        "data/interim",
        "data/reports",

        # Source code directories
        "src/ingest",
        "src/preprocess",
        "src/pricing",
        "src/forecasting",
        "src/dashboards",
        "src/api",
        "src/llm",
        "src/utils",
        "src/reports",
        "src/data_pipeline",

        # Other directories
        "notebooks",
        "models",
        "logs",
        "tests"
    ]

    # Create directories
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")

    # Create empty __init__.py files for Python packages
    init_dirs = [
        "src", "src/ingest", "src/preprocess", "src/pricing", "src/forecasting",
        "src/dashboards", "src/api", "src/llm", "src/utils", "src/reports", "src/data_pipeline"
    ]

    for init_dir in init_dirs:
        init_file = base_dir / init_dir / "__init__.py"
        init_file.touch()
        print(f"Created: {init_file}")

    print("\n✅ Project structure created successfully!")
    print("\n📁 Directory Structure:")
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(str(base_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                print(f"{subindent}{file}")


if __name__ == "__main__":
    create_project_structure()
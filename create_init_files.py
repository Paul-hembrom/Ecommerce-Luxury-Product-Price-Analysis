"""
Script to create ALL required __init__.py files
"""

import os
from pathlib import Path


def create_init_files():
    """Create all required __init__.py files"""
    project_root = Path(__file__).parent
    src_dir = project_root / "src"

    # List of ALL directories that need __init__.py
    directories = [
        src_dir,
        src_dir / "utils",
        src_dir / "ingest",
        src_dir / "preprocess",
        src_dir / "pricing",
        src_dir / "forecasting",
        src_dir / "dashboards",
        src_dir / "api",
        src_dir / "llm",
        src_dir / "reports",
        src_dir / "data_pipeline"
    ]

    created_count = 0
    existing_count = 0

    for directory in directories:
        init_file = directory / "__init__.py"

        if init_file.exists():
            print(f" Already exists: {init_file}")
            existing_count += 1
        else:
            # Create directory if it doesn't exist
            directory.mkdir(parents=True, exist_ok=True)

            # Create empty __init__.py file
            init_file.touch()
            print(f" Created: {init_file}")
            created_count += 1

    print(f"\n Summary: Created {created_count} new files, {existing_count} already existed")
    return created_count + existing_count


if __name__ == "__main__":
    total = create_init_files()
    print(f" Total __init__.py files: {total}")
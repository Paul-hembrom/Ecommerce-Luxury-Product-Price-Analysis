import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingest.load_data import DataLoader
from preprocess.clean_products import DataCleaner


class TestDataPipeline:
    """Test data ingestion and processing pipeline"""

    def test_data_loader_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader("test_path.json")
        assert loader.raw_data_path == "test_path.json"
        assert hasattr(loader, 'processed_dir')

    def test_price_cleaning(self):
        """Test price string to numeric conversion"""
        cleaner = DataCleaner()

        # Test various price formats
        test_cases = [
            ("$555", 555.0),
            ("€500", 500 * 1.07),  # EUR to USD conversion
            ("£400", 400 * 1.22),  # GBP to USD conversion
            ("1000", 1000.0),
            (None, np.nan),
            ("Invalid", np.nan)
        ]

        for price_input, expected in test_cases:
            result = cleaner.clean_price(price_input)
            if np.isnan(expected):
                assert np.isnan(result)
            else:
                assert abs(result - expected) < 0.01

    def test_category_extraction(self):
        """Test category parsing from nested lists"""
        cleaner = DataCleaner()

        test_cases = [
            (["Men", "Shoes", "Sneakers"], {"main_category": "Men", "sub_category": "Shoes"}),
            (["Women", "Bags"], {"main_category": "Women", "sub_category": "Bags"}),
            ([], {"main_category": "Unknown", "sub_category": "Unknown"}),
            (["Accessories"], {"main_category": "Accessories", "sub_category": "Unknown"})
        ]

        for categories_input, expected in test_cases:
            result = cleaner.extract_categories(categories_input)
            assert result == expected

    def test_data_validation(self):
        """Test data validation logic"""
        # Create test data with known issues
        test_data = pd.DataFrame({
            'id': ['1', '2', '3'],
            'brand': ['Gucci', 'Prada', None],
            'price_final': ['$100', 'Invalid', '$200'],
            'price_full': ['$120', '$150', '$250']
        })

        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_product_data(test_data)

        # Assert data quality checks
        assert 'price_final_usd' in cleaned_data.columns
        assert 'brand_clean' in cleaned_data.columns
        assert cleaned_data['price_final_usd'].notna().sum() >= 2  # At least 2 valid prices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
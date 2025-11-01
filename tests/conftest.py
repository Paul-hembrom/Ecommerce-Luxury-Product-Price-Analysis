import pytest
import pandas as pd
import numpy as np
import tempfile
import os

@pytest.fixture
def sample_product_data():
    """Create sample product data for testing"""
    return pd.DataFrame({
        'id': [f'prod_{i}' for i in range(100)],
        'brand': ['Gucci', 'Prada', 'Louis Vuitton', 'Zara'] * 25,
        'name': [f'Product {i}' for i in range(100)],
        'price_final': [f'${i * 100}' for i in range(100)],
        'price_full': [f'${i * 120}' for i in range(100)],
        'discount': [None] * 50 + [f'{i}%' for i in range(50)],
        'currency': ['USD'] * 100,
        'stock_quantity': np.random.randint(0, 500, 100),
        'categories': [['Fashion', 'Bags'], ['Fashion', 'Shoes']] * 50
    })

@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def sample_competitiveness_data():
    """Sample competitiveness analysis results"""
    return pd.DataFrame({
        'brand_clean': ['Gucci', 'Prada', 'Zara'],
        'main_category': ['Bags', 'Bags', 'Bags'],
        'price_positioning': ['Premium', 'Premium', 'Value'],
        'competitiveness_score': [85, 78, 92],
        'optimization_opportunity': ['Maintain', 'Increase Price', 'Decrease Price']
    })
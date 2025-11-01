import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pricing.price_competitiveness import PriceCompetitivenessAnalyzer
from pricing.elasticity_model import PriceElasticityModel


class TestPricingModels:
    """Test pricing analytics models"""

    def setup_method(self):
        """Create test data for pricing models"""
        self.sample_data = pd.DataFrame({
            'brand_clean': ['Gucci', 'Gucci', 'Prada', 'Prada', 'Zara', 'Zara'],
            'main_category': ['Bags', 'Shoes', 'Bags', 'Shoes', 'Bags', 'Shoes'],
            'price_final_usd': [1000, 800, 1200, 900, 200, 150],
            'discount_pct': [0, 10, 5, 15, 20, 25],
            'stock_quantity': [50, 100, 30, 80, 200, 150]
        })

    def test_competitiveness_analysis(self):
        """Test price competitiveness calculations"""
        analyzer = PriceCompetitivenessAnalyzer()
        results = analyzer.calculate_price_positioning(self.sample_data)

        # Check required columns exist
        required_columns = ['brand_clean', 'main_category', 'price_positioning', 'competitiveness_score']
        for col in required_columns:
            assert col in results.columns

        # Check competitiveness scores are within bounds
        assert results['competitiveness_score'].between(0, 100).all()

        # Check price positioning categories
        valid_positions = ['Premium', 'Competitive', 'Value']
        assert all(pos in valid_positions for pos in results['price_positioning'].unique())

    def test_elasticity_model_initialization(self):
        """Test elasticity model setup"""
        model = PriceElasticityModel()
        assert model.elasticity_coefficients == {}
        assert not model.is_trained

    def test_elasticity_dataset_creation(self):
        """Test elasticity dataset preparation"""
        model = PriceElasticityModel()
        elasticity_df = model.create_elasticity_dataset(self.sample_data)

        # Check dataset structure
        expected_columns = ['brand', 'category', 'avg_price', 'demand_proxy', 'log_price', 'log_demand']
        for col in expected_columns:
            assert col in elasticity_df.columns

        # Check data validity
        assert len(elasticity_df) > 0
        assert elasticity_df['avg_price'].notna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
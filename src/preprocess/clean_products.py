import pandas as pd
import numpy as np
import re
from typing import Dict, Any
import logging
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DataCleaner:
    def __init__(self):
        self.currency_rates = {'USD': 1.0, 'EUR': 1.07, 'GBP': 1.22}  # Example rates

    def clean_price(self, price_str: str) -> float:
        """Extract numeric price from string and convert to USD"""
        if pd.isna(price_str) or price_str is None:
            return np.nan

        # Extract numeric value
        price_match = re.search(r'[\d,]+\.?\d*', str(price_str))
        if not price_match:
            return np.nan

        price_val = float(price_match.group().replace(',', ''))

        # Simple currency conversion (in production, use real-time rates)
        if '€' in str(price_str) or 'EUR' in str(price_str):
            return price_val * self.currency_rates['EUR']
        elif '£' in str(price_str) or 'GBP' in str(price_str):
            return price_val * self.currency_rates['GBP']
        else:
            return price_val

    def extract_categories(self, categories_list: list) -> Dict[str, str]:
        """Extract category information"""
        if not categories_list:
            return {'main_category': 'Unknown', 'sub_category': 'Unknown'}

        # Join categories and extract main/sub categories
        categories_text = ' > '.join(categories_list)
        parts = categories_text.split(' > ')

        return {
            'main_category': parts[0] if len(parts) > 0 else 'Unknown',
            'sub_category': parts[1] if len(parts) > 1 else 'Unknown'
        }

    def clean_product_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main data cleaning pipeline"""
        logger.info("Starting data cleaning process")

        # Create a copy to avoid modifying original
        df_clean = df.copy()

        # Clean prices
        df_clean['price_final_usd'] = df_clean['price_final'].apply(self.clean_price)
        df_clean['price_full_usd'] = df_clean['price_full'].apply(self.clean_price)

        # Calculate discount percentage
        df_clean['discount_pct'] = np.where(
            df_clean['price_full_usd'] > 0,
            ((df_clean['price_full_usd'] - df_clean['price_final_usd']) / df_clean['price_full_usd']) * 100,
            0
        )

        # Extract categories
        category_data = df_clean['categories'].apply(self.extract_categories)
        df_clean['main_category'] = category_data.apply(lambda x: x['main_category'])
        df_clean['sub_category'] = category_data.apply(lambda x: x['sub_category'])

        # Clean brand names
        df_clean['brand_clean'] = df_clean['brand'].str.strip().str.title()

        # Handle missing values
        df_clean['stock_quantity'] = df_clean['stock_quantity'].fillna(0)
        df_clean['description'] = df_clean['description'].fillna('No description')

        # Add derived features
        df_clean['has_discount'] = df_clean['discount_pct'] > 0
        df_clean['price_segment'] = pd.cut(
            df_clean['price_final_usd'],
            bins=[0, 100, 300, 600, 1000, float('inf')],
            labels=['Budget', 'Affordable', 'Mid-Range', 'Premium', 'Luxury']
        )

        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean


def main():
    """Main preprocessing pipeline"""
    from ingest.load_data import DataLoader

    # Load data
    loader = DataLoader("D:/Farfetch/farfetch_men_allclothing_products1.json")
    df = loader.load_json_data()

    # Clean data
    cleaner = DataCleaner()
    df_clean = cleaner.clean_product_data(df)

    # Save cleaned data
    output_path = "data/processed/products_cleaned.parquet"
    df_clean.to_parquet(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")

    return df_clean


if __name__ == "__main__":
    df_clean = main()
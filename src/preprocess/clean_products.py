import pandas as pd
import numpy as np
import re
from typing import Dict, Any
import logging
from utils.logger import setup_logger, DataQualityLogger  # Updated imports

logger = setup_logger(__name__)


class DataCleaner:
    def __init__(self):
        self.currency_rates = {'USD': 1.0, 'EUR': 1.07, 'GBP': 1.22}  # Example rates
        self.dq_logger = DataQualityLogger("data_cleaner")

    def clean_price(self, price_str: str) -> float:
        """Extract numeric price from string and convert to USD"""
        if pd.isna(price_str) or price_str is None:
            return np.nan

        try:
            # Extract numeric value
            price_match = re.search(r'[\d,]+\.?\d*', str(price_str))
            if not price_match:
                logger.debug(f"Could not extract numeric price from: {price_str}")
                return np.nan

            price_val = float(price_match.group().replace(',', ''))

            # Simple currency conversion (in production, use real-time rates)
            if '€' in str(price_str) or 'EUR' in str(price_str):
                converted_price = price_val * self.currency_rates['EUR']
                logger.debug(f"Converted {price_val} EUR to {converted_price} USD")
                return converted_price
            elif '£' in str(price_str) or 'GBP' in str(price_str):
                converted_price = price_val * self.currency_rates['GBP']
                logger.debug(f"Converted {price_val} GBP to {converted_price} USD")
                return converted_price
            else:
                return price_val

        except Exception as e:
            logger.warning(f"Error cleaning price '{price_str}': {str(e)}")
            return np.nan

    def extract_categories(self, categories_list: list) -> Dict[str, str]:
        """Extract category information"""
        if not categories_list:
            return {'main_category': 'Unknown', 'sub_category': 'Unknown'}

        try:
            # Join categories and extract main/sub categories
            categories_text = ' > '.join(categories_list)
            parts = categories_text.split(' > ')

            result = {
                'main_category': parts[0] if len(parts) > 0 else 'Unknown',
                'sub_category': parts[1] if len(parts) > 1 else 'Unknown'
            }

            logger.debug(f"Extracted categories: {result} from {categories_list}")
            return result

        except Exception as e:
            logger.warning(f"Error extracting categories from {categories_list}: {str(e)}")
            return {'main_category': 'Unknown', 'sub_category': 'Unknown'}

    def clean_product_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main data cleaning pipeline"""
        logger.info("Starting data cleaning process")

        # Log initial data state
        initial_rows = len(df)
        self.dq_logger.log_data_summary({
            'initial_rows': initial_rows,
            'initial_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum()
        })

        # Create a copy to avoid modifying original
        df_clean = df.copy()

        # Clean prices
        logger.info("Cleaning price columns...")
        df_clean['price_final_usd'] = df_clean['price_final'].apply(self.clean_price)
        df_clean['price_full_usd'] = df_clean['price_full'].apply(self.clean_price)

        # Log price cleaning results
        valid_prices = df_clean['price_final_usd'].notna().sum()
        logger.info(f"Successfully cleaned {valid_prices}/{len(df_clean)} prices")

        # Calculate discount percentage
        df_clean['discount_pct'] = np.where(
            df_clean['price_full_usd'] > 0,
            ((df_clean['price_full_usd'] - df_clean['price_final_usd']) / df_clean['price_full_usd']) * 100,
            0
        )

        # Extract categories
        logger.info("Extracting category information...")
        category_data = df_clean['categories'].apply(self.extract_categories)
        df_clean['main_category'] = category_data.apply(lambda x: x['main_category'])
        df_clean['sub_category'] = category_data.apply(lambda x: x['sub_category'])

        # Clean brand names
        df_clean['brand_clean'] = df_clean['brand'].str.strip().str.title()

        # Handle missing values
        initial_stock_missing = df_clean['stock_quantity'].isna().sum()
        df_clean['stock_quantity'] = df_clean['stock_quantity'].fillna(0)
        df_clean['description'] = df_clean['description'].fillna('No description')

        # Log missing value handling
        if initial_stock_missing > 0:
            logger.info(f"Filled {initial_stock_missing} missing stock quantities with 0")

        # Add derived features
        df_clean['has_discount'] = df_clean['discount_pct'] > 0
        df_clean['price_segment'] = pd.cut(
            df_clean['price_final_usd'],
            bins=[0, 100, 300, 600, 1000, float('inf')],
            labels=['Budget', 'Affordable', 'Mid-Range', 'Premium', 'Luxury']
        )

        # Data quality checks
        self._perform_data_quality_checks(df_clean)

        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")

        # Log final data state
        self.dq_logger.log_data_summary({
            'final_rows': len(df_clean),
            'final_columns': len(df_clean.columns),
            'unique_brands': df_clean['brand_clean'].nunique(),
            'unique_categories': df_clean['main_category'].nunique(),
            'products_with_discount': df_clean['has_discount'].sum()
        })

        return df_clean

    def _perform_data_quality_checks(self, df: pd.DataFrame):
        """Perform comprehensive data quality checks"""
        logger.info("Performing data quality checks...")

        # Check for missing values in critical columns
        critical_columns = ['id', 'brand_clean', 'price_final_usd']
        for col in critical_columns:
            missing_count = df[col].isna().sum()
            self.dq_logger.log_missing_values(col, missing_count, len(df))

        # Check price validity
        negative_prices = (df['price_final_usd'] < 0).sum()
        if negative_prices > 0:
            logger.warning(f"Found {negative_prices} products with negative prices")

        # Check discount validity
        invalid_discounts = ((df['discount_pct'] < 0) | (df['discount_pct'] > 100)).sum()
        if invalid_discounts > 0:
            logger.warning(f"Found {invalid_discounts} products with invalid discounts")

        # Check category distribution
        category_counts = df['main_category'].value_counts()
        logger.info(f"Category distribution: {category_counts.to_dict()}")


def main():
    """Main preprocessing pipeline"""
    from ingest.load_data import DataLoader

    try:
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

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    df_clean = main()
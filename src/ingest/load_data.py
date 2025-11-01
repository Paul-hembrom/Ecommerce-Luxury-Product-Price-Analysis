import pandas as pd
import json
import os
from pathlib import Path
import logging
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_json_data(self) -> pd.DataFrame:
        """Load and parse JSON data from local file"""
        try:
            logger.info(f"Loading data from {self.raw_data_path}")

            with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert to DataFrame
            df = pd.DataFrame(data)
            logger.info(f"Successfully loaded {len(df)} products")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def save_processed_data(self, df: pd.DataFrame, format: str = 'parquet'):
        """Save processed data in specified format"""
        try:
            output_path = self.processed_dir / f"products_cleaned.{format}"

            if format == 'parquet':
                df.to_parquet(output_path, index=False)
            elif format == 'csv':
                df.to_csv(output_path, index=False)
            else:
                raise ValueError("Format must be 'parquet' or 'csv'")

            logger.info(f"Data saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise


def main():
    """Main data ingestion pipeline"""
    raw_data_path = "D:/Farfetch/farfetch_men_allclothing_products1.json"

    loader = DataLoader(raw_data_path)

    # Load data
    df = loader.load_json_data()

    # Display basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head(2)}")

    # Save processed data
    loader.save_processed_data(df, 'parquet')


if __name__ == "__main__":
    main()
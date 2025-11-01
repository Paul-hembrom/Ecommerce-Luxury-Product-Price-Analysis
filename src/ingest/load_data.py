import pandas as pd
import json
import os
from pathlib import Path
import logging
from utils.logger import setup_logger  # Updated import

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

            # Check if file exists
            if not os.path.exists(self.raw_data_path):
                logger.error(f"Data file not found: {self.raw_data_path}")
                raise FileNotFoundError(f"Data file not found: {self.raw_data_path}")

            with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Log data characteristics
            logger.info(f"Successfully loaded {len(df)} products")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")

            return df

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {self.raw_data_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
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
            logger.info(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            return output_path

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise


def main():
    """Main data ingestion pipeline"""
    raw_data_path = "D:/Farfetch/farfetch_men_allclothing_products1.json"

    loader = DataLoader(raw_data_path)

    try:
        # Load data
        df = loader.load_json_data()

        # Display basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample data:\n{df.head(2)}")

        # Save processed data
        output_path = loader.save_processed_data(df, 'parquet')
        logger.info(f"Data pipeline completed successfully. Output: {output_path}")

        return df

    except Exception as e:
        logger.error(f"Data pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
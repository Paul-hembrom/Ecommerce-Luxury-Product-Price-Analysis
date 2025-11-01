import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PriceCompetitivenessAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=5, random_state=42)

    def calculate_price_positioning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price positioning metrics"""
        logger.info("Calculating price positioning metrics")

        # Group by brand and category
        brand_category_stats = df.groupby(['brand_clean', 'main_category']).agg({
            'price_final_usd': ['mean', 'std', 'count'],
            'discount_pct': 'mean',
            'stock_quantity': 'sum'
        }).round(2)

        brand_category_stats.columns = ['_'.join(col).strip() for col in brand_category_stats.columns.values]
        brand_category_stats = brand_category_stats.reset_index()

        # Calculate overall market benchmarks
        market_benchmarks = df.groupby('main_category')['price_final_usd'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).round(2).reset_index()
        market_benchmarks.columns = ['main_category', 'market_mean', 'market_median', 'market_std', 'market_min',
                                     'market_max']

        # Merge benchmarks
        df_analysis = pd.merge(brand_category_stats, market_benchmarks, on='main_category', how='left')

        # Calculate competitiveness metrics
        df_analysis['price_premium_pct'] = ((df_analysis['price_final_usd_mean'] - df_analysis['market_mean']) /
                                            df_analysis['market_mean']) * 100
        df_analysis['competitiveness_score'] = 100 - abs(df_analysis['price_premium_pct'])
        df_analysis['competitiveness_score'] = np.clip(df_analysis['competitiveness_score'], 0, 100)

        # Categorize positioning
        conditions = [
            df_analysis['price_premium_pct'] < -10,
            (df_analysis['price_premium_pct'] >= -10) & (df_analysis['price_premium_pct'] <= 10),
            df_analysis['price_premium_pct'] > 10
        ]
        choices = ['Value', 'Competitive', 'Premium']
        df_analysis['price_positioning'] = np.select(conditions, choices, default='Competitive')

        logger.info("Price positioning analysis completed")
        return df_analysis

    def identify_opportunities(self, df_analysis: pd.DataFrame) -> pd.DataFrame:
        """Identify pricing optimization opportunities"""

        # High price premium with low stock turnover opportunity
        df_analysis['optimization_opportunity'] = np.where(
            (df_analysis['price_premium_pct'] > 20) & (df_analysis['stock_quantity_sum'] > 50),
            'Consider Price Reduction',
            np.where(
                (df_analysis['price_premium_pct'] < -15) & (df_analysis['stock_quantity_sum'] < 20),
                'Potential Price Increase',
                'Maintain Current Pricing'
            )
        )

        return df_analysis


def main():
    """Run price competitiveness analysis"""
    try:
        # Load cleaned data
        df_clean = pd.read_parquet("data/processed/products_cleaned.parquet")

        # Analyze price competitiveness
        analyzer = PriceCompetitivenessAnalyzer()
        df_analysis = analyzer.calculate_price_positioning(df_clean)
        df_opportunities = analyzer.identify_opportunities(df_analysis)

        # Save results
        output_path = "data/processed/price_competitiveness_analysis.parquet"
        df_opportunities.to_parquet(output_path, index=False)
        logger.info(f"Analysis saved to {output_path}")

        return df_opportunities

    except Exception as e:
        logger.error(f"Error in price competitiveness analysis: {str(e)}")
        raise


if __name__ == "__main__":
    df_analysis = main()
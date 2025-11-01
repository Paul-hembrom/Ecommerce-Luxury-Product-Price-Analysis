import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DiscountRecommender:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for discount recommendation model"""

        # Feature engineering
        features = pd.DataFrame()

        # Brand features
        brand_stats = df.groupby('brand_clean').agg({
            'price_final_usd': ['mean', 'std'],
            'discount_pct': 'mean',
            'stock_quantity': 'sum'
        }).round(2)
        brand_stats.columns = ['brand_' + '_'.join(col).strip() for col in brand_stats.columns.values]
        brand_stats = brand_stats.reset_index()

        features = pd.merge(df[['brand_clean', 'main_category', 'price_final_usd', 'stock_quantity']],
                            brand_stats, on='brand_clean', how='left')

        # Category features
        category_stats = df.groupby('main_category').agg({
            'price_final_usd': ['mean', 'std'],
            'discount_pct': 'mean'
        }).round(2)
        category_stats.columns = ['category_' + '_'.join(col).strip() for col in category_stats.columns.values]
        category_stats = category_stats.reset_index()

        features = pd.merge(features, category_stats, on='main_category', how='left')

        # Price segment features
        features['price_ratio_to_brand_mean'] = features['price_final_usd'] / features['brand_price_final_usd_mean']
        features['price_ratio_to_category_mean'] = features['price_final_usd'] / features[
            'category_price_final_usd_mean']
        features['stock_ratio'] = features['stock_quantity'] / features['brand_stock_quantity_sum']

        # Fill NaN values
        features = features.fillna(0)

        return features

    def train_model(self, df: pd.DataFrame, target_col: str = 'discount_pct'):
        """Train discount recommendation model"""
        logger.info("Training discount recommendation model")

        # Prepare features
        features = self.prepare_features(df)
        X = features.select_dtypes(include=[np.number])
        y = df[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Model trained - MAE: {mae:.2f}, R2: {r2:.2f}")
        self.is_trained = True

        return self.model

    def recommend_discounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate discount recommendations"""
        if not self.is_trained:
            self.train_model(df)

        # Prepare features for prediction
        features = self.prepare_features(df)
        X = features.select_dtypes(include=[np.number])

        # Predict optimal discounts
        recommended_discounts = self.model.predict(X)

        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'product_id': df['id'],
            'brand': df['brand_clean'],
            'current_price': df['price_final_usd'],
            'current_discount': df['discount_pct'],
            'recommended_discount': np.clip(recommended_discounts, 0, 70),  # Cap at 70%
            'stock_quantity': df['stock_quantity']
        })

        # Add recommendation rationale
        recommendations['recommendation'] = np.where(
            recommendations['recommended_discount'] > recommendations['current_discount'] + 5,
            'Increase Discount',
            np.where(
                recommendations['recommended_discount'] < recommendations['current_discount'] - 5,
                'Decrease Discount',
                'Maintain Current Discount'
            )
        )

        # Calculate expected price after recommended discount
        recommendations['recommended_price'] = recommendations['current_price'] * (
                    1 - recommendations['recommended_discount'] / 100)

        logger.info("Discount recommendations generated")
        return recommendations


def main():
    """Run discount recommendation analysis"""
    try:
        # Load cleaned data
        df_clean = pd.read_parquet("data/processed/products_cleaned.parquet")

        # Filter products with valid prices and discounts
        df_valid = df_clean[
            (df_clean['price_final_usd'] > 0) &
            (df_clean['price_final_usd'] < 10000) &  # Remove outliers
            (df_clean['discount_pct'] >= 0) &
            (df_clean['discount_pct'] <= 80)
            ].copy()

        # Generate recommendations
        recommender = DiscountRecommender()
        recommendations = recommender.recommend_discounts(df_valid)

        # Save results
        output_path = "data/processed/discount_recommendations.parquet"
        recommendations.to_parquet(output_path, index=False)
        logger.info(f"Recommendations saved to {output_path}")

        return recommendations

    except Exception as e:
        logger.error(f"Error in discount recommendation: {str(e)}")
        raise


if __name__ == "__main__":
    recommendations = main()
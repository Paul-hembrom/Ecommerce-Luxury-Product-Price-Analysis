import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
import logging
from src.utils.logger import setup_logger
import joblib  # Added for pickle saving
import pickle  # Alternative method

logger = setup_logger(__name__)


class PriceElasticityModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.elasticity_coefficients = {}
        self.is_trained = False
        self.feature_columns = None

    def create_elasticity_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dataset for price elasticity analysis"""
        logger.info("Creating price elasticity dataset")

        # Group by brand and category to analyze price-demand relationships
        brand_category_data = []

        for (brand, category), group in df.groupby(['brand_clean', 'main_category']):
            if len(group) < 10:  # Skip small groups
                continue

            # Calculate metrics for elasticity analysis
            avg_price = group['price_final_usd'].mean()
            avg_discount = group['discount_pct'].mean()
            total_stock = group['stock_quantity'].sum()
            product_count = len(group)

            # Use stock quantity as proxy for demand (in real scenario, use actual sales)
            # Higher stock turnover = higher implied demand
            demand_proxy = total_stock / product_count if product_count > 0 else 0

            # Calculate price variance and other features
            price_std = group['price_final_usd'].std()
            price_range = group['price_final_usd'].max() - group['price_final_usd'].min()

            brand_category_data.append({
                'brand': brand,
                'category': category,
                'avg_price': avg_price,
                'avg_discount': avg_discount,
                'demand_proxy': demand_proxy,
                'price_std': price_std,
                'price_range': price_range,
                'product_count': product_count,
                'total_stock': total_stock,
                'log_price': np.log(avg_price + 1),
                'log_demand': np.log(demand_proxy + 1)
            })

        elasticity_df = pd.DataFrame(brand_category_data)
        logger.info(f"Created elasticity dataset with {len(elasticity_df)} brand-category combinations")
        return elasticity_df

    def calculate_elasticity(self, df: pd.DataFrame, method: str = 'regression') -> dict:
        """Calculate price elasticity using different methods"""
        results = {}

        if method == 'regression':
            # Using log-log regression for elasticity
            X = df[['log_price', 'avg_discount', 'price_std']].fillna(0)
            X = sm.add_constant(X)  # Add constant for intercept
            y = df['log_demand']

            # Remove infinite values
            valid_idx = ~(np.isinf(X).any(axis=1) | np.isinf(y))
            X = X[valid_idx]
            y = y[valid_idx]

            if len(X) > 10:  # Ensure sufficient data
                try:
                    model = sm.OLS(y, X).fit()
                    price_elasticity = model.params['log_price']

                    results = {
                        'elasticity': price_elasticity,
                        'p_value': model.pvalues['log_price'],
                        'r_squared': model.rsquared,
                        'confidence_interval': model.conf_int().loc['log_price'].tolist(),
                        'model_summary': str(model.summary())
                    }
                except Exception as e:
                    logger.warning(f"Regression failed: {e}")
                    return {}

        elif method == 'segment_based':
            # Segment-based elasticity calculation
            df_segment = df.copy()
            df_segment['price_quartile'] = pd.qcut(df_segment['avg_price'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

            segment_elasticity = {}
            for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
                segment_data = df_segment[df_segment['price_quartile'] == quartile]
                if len(segment_data) > 5:
                    # Simple correlation-based elasticity
                    correlation = segment_data['avg_price'].corr(segment_data['demand_proxy'])
                    segment_elasticity[quartile] = correlation

            avg_elasticity = np.mean(list(segment_elasticity.values())) if segment_elasticity else 0

            results = {
                'elasticity': avg_elasticity,
                'segment_elasticities': segment_elasticity,
                'method': 'segment_based'
            }

        return results

    def train_elasticity_model(self, df: pd.DataFrame):
        """Train comprehensive elasticity model"""
        logger.info("Training price elasticity model")

        # Create elasticity dataset
        elasticity_df = self.create_elasticity_dataset(df)

        if len(elasticity_df) == 0:
            raise ValueError("Insufficient data for elasticity analysis")

        # Prepare features for machine learning model
        self.feature_columns = ['avg_price', 'avg_discount', 'price_std', 'price_range', 'product_count']
        X = elasticity_df[self.feature_columns].fillna(0)
        y = elasticity_df['demand_proxy']

        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)

        # Scale features
        self.scalers['features'] = StandardScaler()
        X_scaled = self.scalers['features'].fit_transform(X)

        # Train multiple models
        models = {
            'linear': LinearRegression(),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }

        best_score = -np.inf
        best_model = None
        best_model_name = None

        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                avg_score = np.mean(cv_scores)

                logger.info(f"{name} model - Cross-validation R²: {avg_score:.3f}")

                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_model_name = name
            except Exception as e:
                logger.warning(f"Model {name} failed during CV: {e}")
                continue

        if best_model is None:
            raise ValueError("All models failed during training")

        # Train best model on full dataset
        best_model.fit(X_scaled, y)
        self.models['elasticity'] = best_model
        self.models['best_model_name'] = best_model_name

        # Calculate feature importance for interpretation
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, best_model.feature_importances_))
        elif hasattr(best_model, 'coef_'):
            feature_importance = dict(zip(self.feature_columns, best_model.coef_))
        else:
            feature_importance = {col: 0 for col in self.feature_columns}

        # Calculate elasticity for each brand-category combination
        elasticity_results = {}
        for _, row in elasticity_df.iterrows():
            brand_category = f"{row['brand']}_{row['category']}"

            # Calculate regression-based elasticity
            brand_data = elasticity_df[elasticity_df['brand'] == row['brand']]
            if len(brand_data) > 5:  # Only calculate if sufficient data
                elasticity_result = self.calculate_elasticity(brand_data, method='regression')

                if elasticity_result and 'elasticity' in elasticity_result:
                    elasticity_results[brand_category] = {
                        'elasticity': elasticity_result['elasticity'],
                        'p_value': elasticity_result.get('p_value', 1.0),
                        'avg_price': row['avg_price'],
                        'demand_proxy': row['demand_proxy'],
                        'category': row['category']
                    }

        self.elasticity_coefficients = elasticity_results
        self.is_trained = True

        logger.info(f"Elasticity model trained. Best model: {best_model_name}")
        logger.info(f"Calculated elasticity for {len(elasticity_results)} brand-category combinations")

        return {
            'best_model': best_model_name,
            'cv_score': best_score,
            'feature_importance': feature_importance,
            'elasticity_summary': self.get_elasticity_summary()
        }

    def get_elasticity_summary(self) -> dict:
        """Get summary statistics of elasticity coefficients"""
        if not self.elasticity_coefficients:
            return {}

        elasticities = [v['elasticity'] for v in self.elasticity_coefficients.values()]

        return {
            'mean_elasticity': np.mean(elasticities),
            'median_elasticity': np.median(elasticities),
            'std_elasticity': np.std(elasticities),
            'min_elasticity': np.min(elasticities),
            'max_elasticity': np.max(elasticities),
            'elastic_products': len([e for e in elasticities if e < -1.0]),  # Highly elastic
            'inelastic_products': len([e for e in elasticities if e > -0.2]),  # Relatively inelastic
            'unitary_elastic': len([e for e in elasticities if -1.0 <= e <= -0.2]),
            'total_combinations': len(elasticities)
        }

    def predict_demand_change(self, current_price: float, new_price: float,
                              brand: str, category: str) -> dict:
        """Predict demand change based on price change"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        brand_category = f"{brand}_{category}"

        if brand_category not in self.elasticity_coefficients:
            # Use average elasticity if specific combination not found
            elasticity = self.get_elasticity_summary()['mean_elasticity']
            confidence = 'low'
        else:
            elasticity = self.elasticity_coefficients[brand_category]['elasticity']
            p_value = self.elasticity_coefficients[brand_category]['p_value']
            confidence = 'high' if p_value < 0.05 else 'medium'

        # Calculate price change percentage
        price_change_pct = ((new_price - current_price) / current_price) * 100

        # Calculate predicted demand change using elasticity
        # elasticity = % change in demand / % change in price
        demand_change_pct = elasticity * price_change_pct

        # Calculate predicted new demand
        current_demand = self.elasticity_coefficients.get(brand_category, {}).get('demand_proxy', 100)
        predicted_demand = current_demand * (1 + demand_change_pct / 100)

        # Revenue impact analysis
        current_revenue = current_price * current_demand
        predicted_revenue = new_price * predicted_demand
        revenue_change_pct = ((predicted_revenue - current_revenue) / current_revenue) * 100

        # Recommendation based on elasticity and revenue impact
        if elasticity < -1.5:  # Highly elastic demand
            if price_change_pct < 0:
                recommendation = "Excellent opportunity for price decrease - demand will increase significantly"
            else:
                recommendation = "Warning: Price increase will significantly reduce demand and revenue"
        elif elasticity < -1.0:  # Elastic demand
            if price_change_pct < 0:
                recommendation = "Good opportunity for price decrease - demand will increase"
            else:
                recommendation = "Caution: Price increase may significantly reduce demand"
        elif elasticity > -0.3:  # Inelastic demand
            if price_change_pct > 0:
                recommendation = "Opportunity for price increase - demand will not decrease significantly"
            else:
                recommendation = "Price decrease may not significantly increase demand"
        else:  # Unitary elasticity
            recommendation = "Monitor carefully - demand changes proportionally with price"

        # Add revenue impact to recommendation
        if revenue_change_pct > 5:
            recommendation += " (High revenue potential)"
        elif revenue_change_pct < -5:
            recommendation += " (High revenue risk)"

        return {
            'current_price': current_price,
            'new_price': new_price,
            'price_change_pct': price_change_pct,
            'elasticity': elasticity,
            'predicted_demand_change_pct': demand_change_pct,
            'current_demand': current_demand,
            'predicted_demand': predicted_demand,
            'current_revenue': current_revenue,
            'predicted_revenue': predicted_revenue,
            'revenue_change_pct': revenue_change_pct,
            'recommendation': recommendation,
            'confidence': confidence
        }

    def get_optimization_recommendations(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Get price optimization recommendations based on elasticity"""
        recommendations = []

        for brand_category, data in self.elasticity_coefficients.items():
            if data['p_value'] > 0.1:  # Only use statistically significant elasticities
                continue

            brand, category = brand_category.split('_', 1)
            current_price = data['avg_price']
            elasticity = data['elasticity']

            # Calculate optimal price change based on elasticity
            if elasticity < -1.5:  # Highly elastic - consider price decrease
                optimal_price_change = min(-10, -15 / elasticity)  # More aggressive decrease
                rationale = "Highly elastic demand - significant price decrease can dramatically increase revenue"
            elif elasticity < -1.0:  # Elastic - consider price decrease
                optimal_price_change = min(-5, -10 / elasticity)  # Conservative decrease
                rationale = "Elastic demand - price decrease can increase revenue"
            elif elasticity > -0.3:  # Inelastic - consider price increase
                optimal_price_change = min(15, 8 / abs(elasticty))  # Conservative increase
                rationale = "Inelastic demand - opportunity for price increase with minimal demand loss"
            else:
                continue  # Skip unitary elasticity for optimization

            new_price = current_price * (1 + optimal_price_change / 100)

            # Predict impact
            try:
                impact = self.predict_demand_change(current_price, new_price, brand, category)

                recommendations.append({
                    'brand': brand,
                    'category': category,
                    'current_price': current_price,
                    'recommended_price': new_price,
                    'price_change_pct': optimal_price_change,
                    'elasticity': elasticity,
                    'predicted_demand_change_pct': impact['predicted_demand_change_pct'],
                    'predicted_revenue_change_pct': impact['revenue_change_pct'],
                    'rationale': rationale,
                    'confidence': impact['confidence']
                })
            except Exception as e:
                logger.warning(f"Could not generate recommendation for {brand_category}: {e}")
                continue

        recommendations_df = pd.DataFrame(recommendations)
        if len(recommendations_df) > 0:
            recommendations_df = recommendations_df.nlargest(top_n, 'predicted_revenue_change_pct')

        return recommendations_df

    def save_model(self, model_path: str):
        """Save trained model and coefficients using joblib (better for sklearn models)"""
        if not self.is_trained:
            raise ValueError("No model to save")

        # Create models directory if it doesn't exist
        from pathlib import Path
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Save using joblib (recommended for sklearn models)
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'elasticity_coefficients': self.elasticity_coefficients,
            'is_trained': self.is_trained,
            'feature_columns': self.feature_columns
        }

        try:
            # Method 1: Using joblib (preferred for sklearn)
            joblib.dump(model_data, model_path)
            logger.info(f"Elasticity model saved to {model_path} using joblib")

            # Method 2: Also save as pickle for compatibility
            pickle_path = model_path.replace('.pkl', '_pickle.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Elasticity model also saved to {pickle_path} using pickle")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_path: str):
        """Load trained model and coefficients"""
        try:
            # Try joblib first
            model_data = joblib.load(model_path)
        except:
            # Fall back to pickle
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            except Exception as e:
                raise ValueError(f"Could not load model from {model_path}: {e}")

        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.elasticity_coefficients = model_data['elasticity_coefficients']
        self.is_trained = model_data['is_trained']
        self.feature_columns = model_data.get('feature_columns')

        logger.info(f"Elasticity model loaded from {model_path}")
        logger.info(f"Best model: {self.models.get('best_model_name', 'Unknown')}")


def main():
    """Run price elasticity analysis"""
    try:
        # Load data
        df_clean = pd.read_parquet("data/processed/products_cleaned.parquet")

        # Initialize and train elasticity model
        elasticity_model = PriceElasticityModel()
        training_results = elasticity_model.train_elasticity_model(df_clean)

        # Get optimization recommendations
        recommendations = elasticity_model.get_optimization_recommendations(df_clean, top_n=15)

        # Save model
        model_path = "models/price_elasticity_model.pkl"
        elasticity_model.save_model(model_path)

        # Print summary
        summary = elasticity_model.get_elasticity_summary()
        logger.info("Elasticity Analysis Summary:")
        logger.info(f"Mean Elasticity: {summary['mean_elasticity']:.3f}")
        logger.info(f"Elastic Products: {summary['elastic_products']}")
        logger.info(f"Inelastic Products: {summary['inelastic_products']}")
        logger.info(f"Unitary Elastic: {summary['unitary_elastic']}")
        logger.info(f"Optimization Recommendations: {len(recommendations)}")

        # Display top recommendations
        if len(recommendations) > 0:
            print("\nTop 5 Price Optimization Recommendations:")
            display_cols = ['brand', 'category', 'current_price', 'recommended_price',
                            'predicted_revenue_change_pct', 'confidence']
            print(recommendations[display_cols].head().round(2))

        return {
            'elasticity_model': elasticity_model,
            'recommendations': recommendations,
            'summary': summary
        }

    except Exception as e:
        logger.error(f"Error in elasticity analysis: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
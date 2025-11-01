import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import logging
from src.utils.logger import setup_logger
import joblib
from datetime import datetime, timedelta

logger = setup_logger(__name__)


class PriceForecaster:
    def __init__(self, sequence_length: int = 30, forecast_horizon: int = 14):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False

    def create_synthetic_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic time series data for price forecasting"""
        logger.info("Creating synthetic time series data for price forecasting")

        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Group by brand and category to create realistic patterns
        brand_category_combos = df[['brand_clean', 'main_category']].drop_duplicates()

        all_time_series = []

        for _, (brand, category) in brand_category_combos.iterrows():
            # Get base price for this brand-category combo
            base_data = df[(df['brand_clean'] == brand) & (df['main_category'] == category)]
            if len(base_data) == 0:
                continue

            base_price = base_data['price_final_usd'].mean()
            if pd.isna(base_price) or base_price <= 0:
                continue

            # Create realistic price patterns
            np.random.seed(hash(brand + category) % 10000)  # Deterministic seed

            # Seasonal components
            seasonal = np.sin(2 * np.pi * np.arange(len(dates)) / 365) * 0.15
            trend = np.linspace(0, np.random.uniform(-0.1, 0.1), len(dates))
            noise = np.random.normal(0, 0.05, len(dates))

            # Discount events (seasonal sales)
            discount_events = np.zeros(len(dates))
            for month in [1, 6, 11]:  # Major sale months
                event_start = month * 30 + np.random.randint(-7, 7)
                if event_start < len(dates):
                    duration = np.random.randint(14, 30)
                    discount_events[event_start:event_start + duration] = np.random.uniform(0.1, 0.3)

            # Calculate price series
            price_multiplier = 1 + seasonal + trend + noise - discount_events
            prices = base_price * price_multiplier

            # Create time series entry
            ts_data = pd.DataFrame({
                'date': dates,
                'brand': brand,
                'category': category,
                'price': prices,
                'base_price': base_price,
                'discount_level': discount_events * 100,  # Convert to percentage
                'seasonal_factor': seasonal,
                'trend_factor': trend
            })

            all_time_series.append(ts_data)

        full_ts = pd.concat(all_time_series, ignore_index=True)
        logger.info(f"Created time series with {len(full_ts)} records")
        return full_ts

    def prepare_forecasting_data(self, ts_data: pd.DataFrame, brand: str = None, category: str = None) -> tuple:
        """Prepare data for LSTM price forecasting"""

        # Filter for specific brand/category if provided
        if brand and category:
            filtered_data = ts_data[(ts_data['brand'] == brand) & (ts_data['category'] == category)]
        elif brand:
            filtered_data = ts_data[ts_data['brand'] == brand]
        else:
            filtered_data = ts_data

        if len(filtered_data) == 0:
            raise ValueError("No data available for the specified filters")

        # Sort by date
        filtered_data = filtered_data.sort_values('date')

        # Prepare features and target
        features = filtered_data[['price', 'discount_level', 'seasonal_factor', 'trend_factor']].values
        target = filtered_data['price'].values

        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.price_scaler.fit_transform(target.reshape(-1, 1)).flatten()

        return features_scaled, target_scaled, filtered_data

    def create_sequences(self, features: np.ndarray, target: np.ndarray) -> tuple:
        """Create sequences for LSTM training"""
        X, y = [], []

        for i in range(len(features) - self.sequence_length - self.forecast_horizon + 1):
            X.append(features[i:(i + self.sequence_length)])
            y.append(target[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])

        return np.array(X), np.array(y)

    def build_model(self, input_shape: tuple) -> Sequential:
        """Build bidirectional LSTM model for price forecasting"""
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(32, return_sequences=True)),
            Dropout(0.3),
            LSTM(16),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.forecast_horizon)
        ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        logger.info("Bidirectional LSTM model built successfully")
        return model

    def train_model(self, df: pd.DataFrame, brand: str = None, category: str = None):
        """Train price forecasting model"""
        logger.info("Training price forecasting model")

        # Create time series data
        ts_data = self.create_synthetic_time_series(df)

        # Prepare data
        features, target, filtered_data = self.prepare_forecasting_data(ts_data, brand, category)

        # Create sequences
        X, y = self.create_sequences(features, target)

        if len(X) == 0:
            raise ValueError("Insufficient data for training")

        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Build and train model
        self.model = self.build_model((self.sequence_length, X.shape[2]))

        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1,
            shuffle=False,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
            ]
        )

        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        # Inverse transform predictions
        train_pred_actual = self.price_scaler.inverse_transform(train_pred.reshape(-1, 1)).reshape(train_pred.shape)
        test_pred_actual = self.price_scaler.inverse_transform(test_pred.reshape(-1, 1)).reshape(test_pred.shape)
        y_train_actual = self.price_scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
        y_test_actual = self.price_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train_actual.flatten(), train_pred_actual.flatten())
        test_mae = mean_absolute_error(y_test_actual.flatten(), test_pred_actual.flatten())
        train_mape = mean_absolute_percentage_error(y_train_actual.flatten(), train_pred_actual.flatten()) * 100
        test_mape = mean_absolute_percentage_error(y_test_actual.flatten(), test_pred_actual.flatten()) * 100

        logger.info(f"Model training completed - Train MAE: ${train_mae:.2f}, Test MAE: ${test_mae:.2f}")
        logger.info(f"Train MAPE: {train_mape:.2f}%, Test MAPE: {test_mape:.2f}%")

        self.is_trained = True
        return history, (train_mae, test_mae, train_mape, test_mape)

    def forecast_prices(self, last_sequence: np.ndarray) -> np.ndarray:
        """Generate price forecasts"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train_model first.")

        # Ensure sequence is the correct length
        if len(last_sequence) != self.sequence_length:
            raise ValueError(f"Sequence length must be {self.sequence_length}")

        # Scale the input sequence
        last_sequence_scaled = self.feature_scaler.transform(last_sequence)

        # Reshape for prediction
        X_pred = last_sequence_scaled.reshape(1, self.sequence_length, -1)

        # Generate forecast
        forecast_scaled = self.model.predict(X_pred)

        # Inverse transform to get actual prices
        forecast = self.price_scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

        return forecast

    def get_forecast_insights(self, forecasts: np.ndarray, current_price: float) -> dict:
        """Generate insights from price forecasts"""
        avg_forecast = np.mean(forecasts)
        forecast_change_pct = ((avg_forecast - current_price) / current_price) * 100
        volatility = np.std(forecasts) / avg_forecast * 100

        if forecast_change_pct > 5:
            trend = "Strong Upward"
            recommendation = "Consider gradual price increases"
        elif forecast_change_pct > 2:
            trend = "Moderate Upward"
            recommendation = "Monitor market and maintain current pricing"
        elif forecast_change_pct < -5:
            trend = "Strong Downward"
            recommendation = "Evaluate competitive positioning and consider promotions"
        elif forecast_change_pct < -2:
            trend = "Moderate Downward"
            recommendation = "Review pricing strategy and competitor moves"
        else:
            trend = "Stable"
            recommendation = "Maintain current pricing strategy"

        return {
            'average_forecast': avg_forecast,
            'forecast_change_pct': forecast_change_pct,
            'volatility_pct': volatility,
            'trend_direction': trend,
            'recommendation': recommendation,
            'confidence_interval': {
                'lower': np.percentile(forecasts, 25),
                'upper': np.percentile(forecasts, 75)
            }
        }

    def save_model(self, model_path: str):
        """Save trained model and scalers"""
        if not self.is_trained or self.model is None:
            raise ValueError("No model to save")

        self.model.save(model_path)
        joblib.dump(self.price_scaler, model_path.replace('.h5', '_price_scaler.pkl'))
        joblib.dump(self.feature_scaler, model_path.replace('.h5', '_feature_scaler.pkl'))
        logger.info(f"Price forecasting model saved to {model_path}")


def main():
    """Run price forecasting pipeline"""
    try:
        # Load data
        df_clean = pd.read_parquet("data/processed/products_cleaned.parquet")

        # Initialize and train forecaster for a specific brand
        forecaster = PriceForecaster(sequence_length=30, forecast_horizon=14)

        # Get top brands for demonstration
        top_brands = df_clean['brand_clean'].value_counts().head(3).index.tolist()

        results = {}
        for brand in top_brands:
            logger.info(f"Training model for brand: {brand}")
            try:
                history, metrics = forecaster.train_model(df_clean, brand=brand)

                # Save model for this brand
                model_path = f"models/price_forecast_{brand.replace(' ', '_').lower()}.h5"
                forecaster.save_model(model_path)

                results[brand] = {
                    'metrics': metrics,
                    'model_path': model_path
                }

            except Exception as e:
                logger.error(f"Failed to train model for {brand}: {str(e)}")
                continue

        logger.info("Price forecasting pipeline completed")
        return results

    except Exception as e:
        logger.error(f"Error in price forecasting: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
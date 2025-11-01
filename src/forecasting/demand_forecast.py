import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from utils.logger import setup_logger
import joblib

logger = setup_logger(__name__)


class DemandForecaster:
    def __init__(self, sequence_length: int = 30, forecast_horizon: int = 7):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.model = None

    def create_sequences(self, data: np.ndarray) -> tuple:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])
        return np.array(X), np.array(y)

    def build_model(self, input_shape: tuple) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(self.forecast_horizon)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def prepare_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series data from product data"""
        # Since we don't have actual time series, we'll create synthetic dates
        # In production, you would use actual sales data with timestamps

        # Create daily aggregated data (synthetic for demo)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

        # Simulate demand patterns
        np.random.seed(42)
        base_demand = 1000
        seasonal_factor = np.sin(2 * np.pi * np.arange(len(dates)) / 365) * 0.3
        trend = np.arange(len(dates)) * 0.01
        noise = np.random.normal(0, 0.1, len(dates))

        demand = base_demand * (1 + seasonal_factor + trend + noise)

        # Create time series dataframe
        ts_data = pd.DataFrame({
            'date': dates,
            'demand': demand,
            'price_index': np.random.uniform(0.8, 1.2, len(dates)),
            'discount_level': np.random.uniform(0, 0.3, len(dates))
        })

        return ts_data

    def train_model(self, df: pd.DataFrame):
        """Train LSTM demand forecasting model"""
        logger.info("Training LSTM demand forecasting model")

        # Prepare time series data
        ts_data = self.prepare_time_series_data(df)

        # Use demand as target variable
        data = ts_data[['demand', 'price_index', 'discount_level']].values

        # Scale data
        data_scaled = self.scaler.fit_transform(data)

        # Create sequences
        X, y = self.create_sequences(data_scaled[:, 0])  # Using only demand for prediction

        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Build and train model
        self.model = self.build_model((self.sequence_length, 1))

        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1,
            shuffle=False
        )

        # Evaluate model
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        logger.info(f"Model training completed - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return history

    def forecast_demand(self, last_sequence: np.ndarray) -> np.ndarray:
        """Generate demand forecast"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")

        # Scale input sequence
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))[:, 0]

        # Reshape for prediction
        X_pred = last_sequence_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)

        # Generate forecast
        forecast_scaled = self.model.predict(X_pred)

        # Inverse transform
        forecast = self.scaler.inverse_transform(
            np.column_stack([forecast_scaled.reshape(-1, 1), np.zeros((self.forecast_horizon, 2))])
        )[:, 0]

        return forecast

    def save_model(self, model_path: str):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(model_path)
        joblib.dump(self.scaler, model_path.replace('.h5', '_scaler.pkl'))
        logger.info(f"Model saved to {model_path}")


def main():
    """Run demand forecasting pipeline"""
    try:
        # Load data
        df_clean = pd.read_parquet("data/processed/products_cleaned.parquet")

        # Initialize and train forecaster
        forecaster = DemandForecaster(sequence_length=30, forecast_horizon=7)
        history = forecaster.train_model(df_clean)

        # Save model
        model_path = "models/demand_forecast_model.h5"
        forecaster.save_model(model_path)

        logger.info("Demand forecasting pipeline completed")
        return forecaster

    except Exception as e:
        logger.error(f"Error in demand forecasting: {str(e)}")
        raise


if __name__ == "__main__":
    forecaster = main()
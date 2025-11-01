"""
Utility for loading saved models including .pkl files
"""
import joblib
import pickle
import logging
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelLoader:
    @staticmethod
    def load_model(model_path: str):
        """Load a saved model, trying different methods"""
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Try joblib first (better for sklearn models)
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully using joblib: {model_path}")
            return model
        except Exception as e:
            logger.warning(f"Joblib loading failed, trying pickle: {e}")
            try:
                # Try pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Model loaded successfully using pickle: {model_path}")
                return model
            except Exception as e2:
                logger.error(f"Both joblib and pickle loading failed: {e2}")
                raise

    @staticmethod
    def load_elasticity_model(model_path: str = "models/price_elasticity_model.pkl"):
        """Load the price elasticity model specifically"""
        from pricing.elasticity_model import PriceElasticityModel

        try:
            # Load the model data
            model_data = ModelLoader.load_model(model_path)

            # Create a new elasticity model instance
            elasticity_model = PriceElasticityModel()

            # Restore the state
            elasticity_model.models = model_data['models']
            elasticity_model.scalers = model_data['scalers']
            elasticity_model.elasticity_coefficients = model_data['elasticity_coefficients']
            elasticity_model.is_trained = model_data['is_trained']
            elasticity_model.feature_columns = model_data.get('feature_columns')

            logger.info(f"Elasticity model loaded successfully from {model_path}")
            return elasticity_model

        except Exception as e:
            logger.error(f"Failed to load elasticity model: {e}")
            raise


def test_model_loading():
    """Test function to verify model loading works"""
    try:
        # Test loading elasticity model
        elasticity_model = ModelLoader.load_elasticity_model()
        print("✅ Elasticity model loaded successfully!")

        # Print model info
        summary = elasticity_model.get_elasticity_summary()
        print(f"Model summary: {summary}")

        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


if __name__ == "__main__":
    test_model_loading()
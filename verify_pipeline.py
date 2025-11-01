"""
Script to verify that all required files are generated correctly
"""
import pandas as pd
from pathlib import Path
import json


def verify_models():
    """Verify that ML models are generated including .pkl files"""
    print("\n🔍 VERIFYING ML MODELS")
    print("=" * 60)

    models_dir = Path("models")
    expected_models = [
        "demand_forecast_model.h5",
        "price_elasticity_model.pkl",  # This should now exist
        "price_elasticity_model_pickle.pkl"  # Backup pickle file
    ]

    all_good = True

    for model_file in expected_models:
        model_path = models_dir / model_file
        if model_path.exists():
            # Try to load the model to verify it's valid
            try:
                if model_file.endswith('.pkl'):
                    from src.utils.model_loader import ModelLoader
                    if model_file == "price_elasticity_model.pkl":
                        model = ModelLoader.load_elasticity_model(str(model_path))
                        print(f"✅ {model_file}: Valid elasticity model loaded")
                    else:
                        model = ModelLoader.load_model(str(model_path))
                        print(f"✅ {model_file}: Valid pickle model loaded")
                else:
                    # For .h5 files, just check existence
                    print(f"✅ {model_file}: Model file exists")
            except Exception as e:
                print(f"❌ {model_file}: File exists but cannot be loaded - {e}")
                all_good = False
        else:
            # Only require the main .pkl file
            if model_file == "price_elasticity_model.pkl":
                print(f"❌ {model_file}: Main elasticity model not found")
                all_good = False
            else:
                print(f"⚠️  {model_file}: Optional model file not found")

    # Check for price forecast models
    price_forecast_models = list(models_dir.glob("price_forecast_*.h5"))
    if price_forecast_models:
        print(f"✅ Price forecast models: {len(price_forecast_models)} brand-specific models")
    else:
        print("❌ Price forecast models: No brand-specific models found")
        all_good = False

    print("=" * 60)
    return all_good


if __name__ == "__main__":
    data_ok = verify_data_pipeline()
    models_ok = verify_models()

    if data_ok and models_ok:
        print("\n🎉 COMPLETE VERIFICATION: ALL SYSTEMS GO!")
        print("\n🚀 You can now:")
        print("   - Run the dashboard: streamlit run src/dashboards/streamlit_app.py")
        print("   - Start the API: python src/api/api_server.py")
        print("   - Explore the EDA: jupyter notebook notebooks/exploratory_analysis.ipynb")
    else:
        print("\n⚠️  VERIFICATION FAILED: Some components missing")
        print("   Please run the data pipeline first: python main.py")
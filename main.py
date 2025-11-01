"""
Farfetch Pricing Analytics Engine - Updated Main Execution Script
With Complete Data Processing Pipeline and Fixed Imports
"""

import sys
import os
from pathlib import Path

# Setup Python path first
project_root = Path(__file__).parent
src_path = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Now import other modules
try:
    from src.utils.logger import setup_logger
    from src.data_pipeline.process_data import DataProcessingPipeline
    from src.forecasting.demand_forecast import DemandForecaster
    from src.forecasting.price_forecast import PriceForecaster
    from src.reports.generate_report import PricingReportGenerator
    from src.llm.insight_generator import PricingInsightGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative imports...")
    from utils.logger import setup_logger
    from data_pipeline.process_data import DataProcessingPipeline
    from forecasting.demand_forecast import DemandForecaster
    from forecasting.price_forecast import PriceForecaster
    from reports.generate_report import PricingReportGenerator
    from llm.insight_generator import PricingInsightGenerator

logger = setup_logger(__name__)


class CompletePricingEngine:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.results = {}

    def run_data_pipeline(self):
        """Run complete data processing pipeline"""
        logger.info("Step 1: Running Data Processing Pipeline")

        pipeline = DataProcessingPipeline(self.raw_data_path)
        self.results = pipeline.run_complete_pipeline()

        logger.info("    Data pipeline completed successfully")
        return self.results

    def run_ml_forecasting(self):
        """Run machine learning forecasting models"""
        logger.info("Step 2: Running ML Forecasting Models")

        # Demand forecasting
        demand_forecaster = DemandForecaster()
        demand_history = demand_forecaster.train_model(self.results['cleaned_data'])
        demand_forecaster.save_model("models/demand_forecast_model.h5")

        # Price forecasting for top brands
        price_forecaster = PriceForecaster()
        price_results = {}

        top_brands = self.results['cleaned_data']['brand_clean'].value_counts().head(3).index.tolist()

        for brand in top_brands:
            try:
                logger.info(f"Training price forecast model for: {brand}")
                history, metrics = price_forecaster.train_model(self.results['cleaned_data'], brand=brand)
                model_path = f"models/price_forecast_{brand.replace(' ', '_').lower()}.h5"
                price_forecaster.save_model(model_path)
                price_results[brand] = {'metrics': metrics, 'model_path': model_path}
            except Exception as e:
                logger.warning(f"Could not train model for {brand}: {str(e)}")

        logger.info(" ML forecasting completed successfully")
        return {
            'demand_forecaster': demand_forecaster,
            'price_forecasting': price_results
        }

    def generate_insights_and_reports(self):
        """Generate AI insights and comprehensive reports"""
        logger.info("Step 3: Generating Insights and Reports")

        # AI-powered insights
        insight_generator = PricingInsightGenerator()
        insights = insight_generator.get_comprehensive_insights(
            self.results['cleaned_data'],
            self.results['competitiveness_analysis']
        )

        # PDF report generation
        report_generator = PricingReportGenerator()
        report_path = report_generator.generate_pdf_report(
            self.results['cleaned_data'],
            self.results['competitiveness_analysis'],
            self.results['discount_recommendations']
        )

        logger.info(" Insights and reports generated successfully")
        return {
            'ai_insights': insights,
            'report_path': report_path
        }

    def display_final_summary(self):
        """Display final execution summary"""
        print("\n" + "=" * 80)
        print(" FARFETCH PRICING ANALYTICS ENGINE - EXECUTION COMPLETE!")
        print("=" * 80)

        print("\n DATA PROCESSING RESULTS:")
        print(f"   Products Analyzed: {len(self.results['cleaned_data']):,}")
        print(f"   Brands Processed: {self.results['cleaned_data']['brand_clean'].nunique():,}")
        print(f"   Categories Analyzed: {self.results['cleaned_data']['main_category'].nunique():,}")

        print("\n ANALYTICS GENERATED:")
        print(f"   Competitiveness Analysis: {len(self.results['competitiveness_analysis']):,} combinations")
        print(f"   Discount Recommendations: {len(self.results['discount_recommendations']):,} products")
        print(f"   Elasticity Optimizations: {len(self.results['elasticity_recommendations']):,} opportunities")

        print("\n AI & ML MODELS:")
        print("   Demand Forecasting Model: Trained (LSTM)")
        print("   Price Forecasting Models:  Trained for top brands")
        print("   Elasticity Model:  Trained and optimized")
        print("   LLM Insights:  Generated natural language analysis")

        print("\n OUTPUT FILES GENERATED:")
        print("   data/processed/products_cleaned.parquet")
        print("   data/processed/price_competitiveness_analysis.parquet")
        print("   data/processed/discount_recommendations.parquet")
        print("   data/processed/elasticity_results.parquet")
        print("   data/processed/master_pricing_dataset.parquet")
        print("   models/demand_forecast_model.h5")
        print("   models/price_elasticity_model.pkl")
        print("   models/price_forecast_*.h5 (for top brands)")

        print("\n NEXT STEPS:")
        print("   1. Start Dashboard: streamlit run src/dashboards/streamlit_app.py")
        print("   2. Start API: python src/api/api_server.py")
        print("   3. View Reports: Check 'data/reports/' directory")
        print("   4. Explore EDA: Run notebooks/exploratory_analysis.ipynb")

        print("\n" + "=" * 80)

    def run_complete_engine(self):
        """Run complete pricing analytics engine"""
        try:
            logger.info("Starting Complete Farfetch Pricing Analytics Engine")

            # Step 1: Data Processing
            self.run_data_pipeline()

            # Step 2: ML Forecasting
            ml_results = self.run_ml_forecasting()
            self.results.update(ml_results)

            # Step 3: Insights & Reports
            insight_results = self.generate_insights_and_reports()
            self.results.update(insight_results)

            # Final Summary
            self.display_final_summary()

            return self.results

        except Exception as e:
            logger.error(f"Engine execution failed: {str(e)}")
            raise


def start_dashboard():
    """Start Streamlit dashboard"""
    print("Starting Streamlit Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboards/streamlit_app.py"])


def start_api():
    """Start FastAPI server"""
    print("Starting FastAPI Server...")
    uvicorn.run("api.api_server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    raw_data_path = "D:/Farfetch/farfetch_men_allclothing_products1.json"

    engine = CompletePricingEngine(raw_data_path)
    results = engine.run_complete_engine()

    # Optional: Uncomment to start services automatically
    # start_dashboard()
    # start_api()
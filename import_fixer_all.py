"""
Comprehensive import fixer for Farfetch Pricing Analytics
Creates all necessary __init__.py files and tests imports.
"""
import os
import sys
from pathlib import Path


def create_init_files():
    """Create all __init__.py files with proper content"""

    init_files = {
        'src/__init__.py': '''"""
Farfetch Pricing Analytics Engine
A comprehensive pricing analytics platform for luxury e-commerce.
"""

__version__ = "1.0.0"
__author__ = "Paul Hembrom"

# Package imports - import only after everything is loaded
def import_all():
    """Import all submodules - call this after package is fully loaded"""
    from . import ingest
    from . import preprocess
    from . import pricing
    from . import forecasting
    from . import dashboards
    from . import api
    from . import llm
    from . import utils
    from . import reports
    from . import data_pipeline

__all__ = [
    'ingest',
    'preprocess', 
    'pricing',
    'forecasting',
    'dashboards',
    'api', 
    'llm',
    'utils',
    'reports',
    'data_pipeline'
]
''',

        'src/pricing/__init__.py': '''"""
Pricing analytics modules
Price competitiveness, discount recommendations, and elasticity modeling.
"""

# Import classes directly
from .price_competitiveness import PriceCompetitivenessAnalyzer
from .discount_recommendation import DiscountRecommender
from .elasticity_model import PriceElasticityModel

__all__ = [
    'PriceCompetitivenessAnalyzer',
    'DiscountRecommender', 
    'PriceElasticityModel'
]
''',

        'src/utils/__init__.py': '''"""
Utility modules
Logging, helpers, and model loading utilities.
"""

from .logger import setup_logger, get_project_logger, PerformanceLogger, DataQualityLogger
from .helpers import format_currency, calculate_percentage_change, safe_divide
from .model_loader import ModelLoader

__all__ = [
    'setup_logger',
    'get_project_logger',
    'PerformanceLogger', 
    'DataQualityLogger',
    'format_currency',
    'calculate_percentage_change',
    'safe_divide',
    'ModelLoader'
]
''',

        'src/forecasting/__init__.py': '''"""
Forecasting modules
Demand and price forecasting using LSTM models.
"""

from .demand_forecast import DemandForecaster
from .price_forecast import PriceForecaster

__all__ = [
    'DemandForecaster',
    'PriceForecaster'
]
''',

        'src/data_pipeline/__init__.py': '''"""
Data pipeline modules
End-to-end data processing pipeline.
"""

from .process_data import DataProcessingPipeline

__all__ = [
    'DataProcessingPipeline'
]
''',

        'src/ingest/__init__.py': '''"""
Data ingestion modules
"""

from .load_data import DataLoader

__all__ = ['DataLoader']
''',

        'src/preprocess/__init__.py': '''"""
Data preprocessing modules
"""

from .clean_products import DataCleaner

__all__ = ['DataCleaner']
''',

        'src/dashboards/__init__.py': '''"""
Dashboard modules
Streamlit web interface.
"""

from .streamlit_app import PricingDashboard

__all__ = ['PricingDashboard']
''',

        'src/api/__init__.py': '''"""
API modules
FastAPI REST interface.
"""

from .api_server import app

__all__ = ['app']
''',

        'src/llm/__init__.py': '''"""
LLM modules
AI-powered insights generation.
"""

from .insight_generator import PricingInsightGenerator

__all__ = ['PricingInsightGenerator']
''',

        'src/reports/__init__.py': '''"""
Reporting modules
PDF report generation.
"""

from .generate_report import PricingReportGenerator

__all__ = ['PricingReportGenerator']
'''
    }

    created_count = 0
    for file_path, content in init_files.items():
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created: {file_path}")
        created_count += 1

    return created_count


def setup_python_path():
    """Setup Python path correctly"""
    project_root = Path(__file__).parent
    src_path = project_root / "src"

    # Clean existing paths
    sys.path = [p for p in sys.path if str(project_root) not in p and str(src_path) not in p]

    # Add paths
    sys.path.insert(0, str(src_path))
    sys.path.insert(0, str(project_root))

    print(f"✅ Python path configured:")
    print(f"   Project: {project_root}")
    print(f"   Source: {src_path}")

    return project_root, src_path


def test_all_imports():
    """Test all imports after setup"""
    print("\n🧪 Testing all imports...")

    imports_to_test = [
        # Utils
        ("utils.logger", "setup_logger"),
        ("utils.helpers", "format_currency"),
        ("utils.model_loader", "ModelLoader"),

        # Pricing
        ("pricing.price_competitiveness", "PriceCompetitivenessAnalyzer"),
        ("pricing.discount_recommendation", "DiscountRecommender"),
        ("pricing.elasticity_model", "PriceElasticityModel"),

        # Data pipeline
        ("data_pipeline.process_data", "DataProcessingPipeline"),

        # Forecasting
        ("forecasting.demand_forecast", "DemandForecaster"),
        ("forecasting.price_forecast", "PriceForecaster"),

        # Other modules
        ("ingest.load_data", "DataLoader"),
        ("preprocess.clean_products", "DataCleaner"),
        ("dashboards.streamlit_app", "PricingDashboard"),
        ("api.api_server", "app"),
        ("llm.insight_generator", "PricingInsightGenerator"),
        ("reports.generate_report", "PricingReportGenerator"),
    ]

    successful = 0
    failed = []

    for module_path, class_name in imports_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            obj = getattr(module, class_name)
            print(f"✅ {module_path}.{class_name}")
            successful += 1
        except Exception as e:
            print(f"❌ {module_path}.{class_name} - {e}")
            failed.append((module_path, class_name, str(e)))

    print(f"\n📊 Import Test Results:")
    print(f"   ✅ Successful: {successful}/{len(imports_to_test)}")
    print(f"   ❌ Failed: {len(failed)}/{len(imports_to_test)}")

    if failed:
        print("\n⚠️ Failed imports:")
        for module, cls, error in failed:
            print(f"   - {module}.{cls}: {error}")

    return successful == len(imports_to_test)


def create_fixed_main_file():
    """Create a fixed main.py file with proper imports"""
    main_content = '''"""
Farfetch Pricing Analytics Engine - Fixed with Proper Imports
"""
import sys
import os
from pathlib import Path

def setup_environment():
    """Setup Python environment"""
    project_root = Path(__file__).parent
    src_path = project_root / "src"

    # Add to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    print("✅ Environment setup complete")
    return project_root

# Setup environment first
setup_environment()

# Now import using the package structure
try:
    from src.utils.logger import setup_logger
    from src.data_pipeline.process_data import DataProcessingPipeline
    from src.forecasting.demand_forecast import DemandForecaster
    from src.forecasting.price_forecast import PriceForecaster
    from src.reports.generate_report import PricingReportGenerator
    from src.llm.insight_generator import PricingInsightGenerator
    print("✅ All imports successful using package structure!")
except ImportError as e:
    print(f"❌ Package imports failed: {e}")
    print("💡 Trying direct imports...")

    # Fallback to direct imports
    from utils.logger import setup_logger
    from data_pipeline.process_data import DataProcessingPipeline
    from forecasting.demand_forecast import DemandForecaster
    from forecasting.price_forecast import PriceForecaster
    from reports.generate_report import PricingReportGenerator
    from llm.insight_generator import PricingInsightGenerator
    print("✅ Direct imports successful!")

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

        logger.info("✅ Data pipeline completed successfully")
        return self.results

    def run_complete_engine(self):
        """Run complete pricing analytics engine"""
        try:
            logger.info("Starting Complete Farfetch Pricing Analytics Engine")

            # Run data pipeline
            self.run_data_pipeline()

            print("\\n" + "="*80)
            print("🎉 FARFETCH PRICING ANALYTICS ENGINE - EXECUTION COMPLETE!")
            print("="*80)

            print("\\n📊 DATA PROCESSING RESULTS:")
            print(f"   Products Analyzed: {len(self.results['cleaned_data']):,}")
            print(f"   Brands Processed: {self.results['cleaned_data']['brand_clean'].nunique():,}")

            print("\\n🚀 NEXT STEPS:")
            print("   1. Start Dashboard: streamlit run src/dashboards/streamlit_app.py")
            print("   2. Start API: python src/api/api_server.py")

            return self.results

        except Exception as e:
            logger.error(f"Engine execution failed: {str(e)}")
            raise

if __name__ == "__main__":
    raw_data_path = "D:/Farfetch/farfetch_men_allclothing_products1.json"

    # Create sample data if file doesn't exist
    if not os.path.exists(raw_data_path):
        print(f"⚠️  Data file not found: {raw_data_path}")
        print("💡 Creating sample data for testing...")

        os.makedirs("data/raw", exist_ok=True)
        import json
        sample_data = [{
            "id": "1", "brand": "Gucci", "price_final": "$500", 
            "price_full": "$600", "stock_quantity": 100,
            "categories": ["Men", "Shoes"]
        }]

        with open("data/raw/sample_products.json", "w") as f:
            json.dump(sample_data, f, indent=2)

        raw_data_path = "data/raw/sample_products.json"
        print(f"✅ Created sample data: {raw_data_path}")

    engine = CompletePricingEngine(raw_data_path)
    results = engine.run_complete_engine()
'''

    with open("main_fixed.py", "w", encoding="utf-8") as f:
        f.write(main_content)
    print("✅ Created: main_fixed.py")


if __name__ == "__main__":
    print("🔧 Farfetch Pricing Analytics - Comprehensive Import Fixer")
    print("=" * 60)

    # Step 1: Create all __init__.py files
    print("\n📁 Creating package structure...")
    created_count = create_init_files()
    print(f"✅ Created {created_count} __init__.py files")

    # Step 2: Setup Python path
    print("\n🐍 Configuring Python path...")
    project_root, src_path = setup_python_path()

    # Step 3: Test imports
    print("\n🧪 Testing imports with new structure...")
    success = test_all_imports()

    # Step 4: Create fixed main file
    print("\n📄 Creating fixed main file...")
    create_fixed_main_file()

    if success:
        print("\n🎉 ALL IMPORTS FIXED SUCCESSFULLY!")
        print("\n🚀 Now you can run:")
        print("   python main_fixed.py")
        print("   streamlit run src/dashboards/streamlit_app.py")
        print("   python src/api/api_server.py")
    else:
        print("\n⚠️ Some imports still need attention.")
        print("💡 Try running individual modules directly.")
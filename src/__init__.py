"""
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

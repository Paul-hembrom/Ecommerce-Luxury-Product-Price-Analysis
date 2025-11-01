# System Architecture

## High-Level Overview
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Data Sources │───▶│ Processing │───▶│ Analytics │
│ - Farfetch API│ │ Pipeline │ │ Engine │
│ - JSON Files │ │ - Cleaning │ │ - ML Models │
└─────────────────┘ │ - Validation │ │ - Business │
└──────────────────┘ │ Logic │
│ └─────────────────┘
▼ │
┌─────────────────┐ │
│ Data Store │◀─────────────┘
│ - Parquet │
│ - Models │
└─────────────────┘
│
▼
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ End Users │◀───│ API Layer │◀───│ Dashboard │
│ - Analysts │ │ - FastAPI │ │ - Streamlit │
│ - Systems │ │ - REST │ │ - Interactive │
└─────────────────┘ └──────────────────┘ └─────────────────┘

text

## Data Flow

1. **Ingestion**: Raw JSON → Pandas DataFrame
2. **Cleaning**: Standardization, validation, enrichment
3. **Analysis**: ML models, business logic, insights
4. **Storage**: Processed data + serialized models
5. **Serving**: APIs + dashboards for consumption
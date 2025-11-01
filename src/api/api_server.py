from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import setup_logger
from pricing.price_competitiveness import PriceCompetitivenessAnalyzer
from pricing.discount_recommendation import DiscountRecommender

logger = setup_logger(__name__)

app = FastAPI(
    title="Farfetch Pricing Analytics API",
    description="API for pricing analysis and recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class Product(BaseModel):
    id: str
    brand: str
    name: str
    price_final: float
    discount_pct: float
    stock_quantity: int
    main_category: str


class PriceRecommendation(BaseModel):
    product_id: str
    brand: str
    current_price: float
    recommended_price: float
    recommended_discount: float
    recommendation: str
    confidence: float


class CompetitivenessResponse(BaseModel):
    brand: str
    category: str
    price_positioning: str
    competitiveness_score: float
    price_premium_pct: float
    optimization_opportunity: str


# Global data storage
df_clean = None
df_competitiveness = None
df_recommendations = None


@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    global df_clean, df_competitiveness, df_recommendations
    try:
        df_clean = pd.read_parquet("data/processed/products_cleaned.parquet")
        df_competitiveness = pd.read_parquet("data/processed/price_competitiveness_analysis.parquet")
        df_recommendations = pd.read_parquet("data/processed/discount_recommendations.parquet")
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


@app.get("/")
async def root():
    return {"message": "Farfetch Pricing Analytics API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "data_loaded": df_clean is not None}


@app.get("/products/", response_model=List[Product])
async def get_products(
        brand: Optional[str] = Query(None, description="Filter by brand"),
        category: Optional[str] = Query(None, description="Filter by category"),
        min_price: Optional[float] = Query(None, description="Minimum price"),
        max_price: Optional[float] = Query(None, description="Maximum price"),
        limit: int = Query(100, description="Number of products to return")
):
    """Get products with optional filtering"""
    if df_clean is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    filtered_df = df_clean.copy()

    if brand:
        filtered_df = filtered_df[filtered_df['brand_clean'] == brand]
    if category:
        filtered_df = filtered_df[filtered_df['main_category'] == category]
    if min_price is not None:
        filtered_df = filtered_df[filtered_df['price_final_usd'] >= min_price]
    if max_price is not None:
        filtered_df = filtered_df[filtered_df['price_final_usd'] <= max_price]

    products = filtered_df.head(limit).to_dict('records')
    return products


@app.get("/competitiveness/", response_model=List[CompetitivenessResponse])
async def get_competitiveness(
        brand: Optional[str] = Query(None, description="Filter by brand"),
        category: Optional[str] = Query(None, description="Filter by category")
):
    """Get price competitiveness analysis"""
    if df_competitiveness is None:
        raise HTTPException(status_code=500, detail="Competitiveness data not loaded")

    filtered_df = df_competitiveness.copy()

    if brand:
        filtered_df = filtered_df[filtered_df['brand_clean'] == brand]
    if category:
        filtered_df = filtered_df[filtered_df['main_category'] == category]

    response_data = []
    for _, row in filtered_df.iterrows():
        response_data.append(CompetitivenessResponse(
            brand=row['brand_clean'],
            category=row['main_category'],
            price_positioning=row['price_positioning'],
            competitiveness_score=row['competitiveness_score'],
            price_premium_pct=row['price_premium_pct'],
            optimization_opportunity=row['optimization_opportunity']
        ))

    return response_data


@app.get("/recommendations/", response_model=List[PriceRecommendation])
async def get_recommendations(
        brand: Optional[str] = Query(None, description="Filter by brand"),
        min_stock: int = Query(0, description="Minimum stock quantity"),
        limit: int = Query(50, description="Number of recommendations to return")
):
    """Get price and discount recommendations"""
    if df_recommendations is None:
        raise HTTPException(status_code=500, detail="Recommendations data not loaded")

    filtered_df = df_recommendations.copy()

    if brand:
        filtered_df = filtered_df[filtered_df['brand'] == brand]

    filtered_df = filtered_df[filtered_df['stock_quantity'] >= min_stock]

    recommendations = []
    for _, row in filtered_df.head(limit).iterrows():
        # Calculate confidence based on stock and discount delta
        confidence = min(0.95, row['stock_quantity'] / 1000 + 0.5)

        recommendations.append(PriceRecommendation(
            product_id=row['product_id'],
            brand=row['brand'],
            current_price=row['current_price'],
            recommended_price=row['recommended_price'],
            recommended_discount=row['recommended_discount'],
            recommendation=row['recommendation'],
            confidence=confidence
        ))

    return recommendations


@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get overall analytics summary"""
    if df_clean is None or df_competitiveness is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    summary = {
        "total_products": len(df_clean),
        "total_brands": df_clean['brand_clean'].nunique(),
        "total_categories": df_clean['main_category'].nunique(),
        "average_price": float(df_clean['price_final_usd'].mean()),
        "average_discount": float(df_clean['discount_pct'].mean()),
        "products_on_discount": int(df_clean['has_discount'].sum()),
        "average_competitiveness_score": float(df_competitiveness['competitiveness_score'].mean()),
        "premium_brands": len(df_competitiveness[df_competitiveness['price_positioning'] == 'Premium']),
        "optimization_opportunities": len(df_competitiveness[
                                              df_competitiveness[
                                                  'optimization_opportunity'] != 'Maintain Current Pricing'
                                              ])
    }

    return summary


@app.post("/predict/demand/")
async def predict_demand(product_ids: List[str]):
    """Predict demand for specific products (placeholder)"""
    # In production, this would use the trained LSTM model
    predictions = {}
    for product_id in product_ids:
        # Mock prediction - replace with actual model inference
        predictions[product_id] = {
            "predicted_demand": np.random.randint(50, 500),
            "confidence": np.random.uniform(0.7, 0.95)
        }

    return predictions


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
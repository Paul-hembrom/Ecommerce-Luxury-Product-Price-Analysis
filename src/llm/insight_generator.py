import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, Any, List
import logging
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PricingInsightGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.insight_pipeline = None
        self.setup_model()

    def setup_model(self):
        """Setup the lightweight LLM for insights"""
        try:
            logger.info("Loading lightweight LLM for insights...")

            # Use a smaller model for efficiency
            self.insight_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",  # Smaller version for efficiency
                tokenizer="microsoft/DialoGPT-small",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=0 if torch.cuda.is_available() else -1,
                max_length=200,
                do_sample=True,
                temperature=0.7
            )

            logger.info("LLM model loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load LLM model: {str(e)}. Using rule-based insights.")
            self.insight_pipeline = None

    def generate_data_insights(self, df_clean: pd.DataFrame, df_competitiveness: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic data insights for LLM context"""
        insights = {
            "total_products": len(df_clean),
            "total_brands": df_clean['brand_clean'].nunique(),
            "avg_price": df_clean['price_final_usd'].mean(),
            "avg_discount": df_clean['discount_pct'].mean(),
            "premium_brands": len(df_competitiveness[df_competitiveness['price_positioning'] == 'Premium']),
            "value_brands": len(df_competitiveness[df_competitiveness['price_positioning'] == 'Value']),
            "top_category": df_clean['main_category'].value_counts().index[0] if len(df_clean) > 0 else "N/A",
            "optimization_opportunities": len(df_competitiveness[
                                                  df_competitiveness[
                                                      'optimization_opportunity'] != 'Maintain Current Pricing'
                                                  ])
        }
        return insights

    def generate_llm_insight(self, data_insights: Dict[str, Any]) -> str:
        """Generate natural language insight using LLM"""
        if self.insight_pipeline is None:
            return self.generate_rule_based_insight(data_insights)

        try:
            prompt = f"""
            Based on the following pricing analytics data, provide 2-3 key business insights and recommendations:

            - Total Products: {data_insights['total_products']:,}
            - Total Brands: {data_insights['total_brands']:,}
            - Average Price: ${data_insights['avg_price']:.2f}
            - Average Discount: {data_insights['avg_discount']:.1f}%
            - Premium Brands: {data_insights['premium_brands']}
            - Value Brands: {data_insights['value_brands']}
            - Top Category: {data_insights['top_category']}
            - Optimization Opportunities: {data_insights['optimization_opportunities']}

            Key Insights:
            """

            response = self.insight_pipeline(
                prompt,
                max_new_tokens=150,
                num_return_sequences=1,
                pad_token_id=self.insight_pipeline.tokenizer.eos_token_id
            )

            insight = response[0]['generated_text'].replace(prompt, '').strip()
            return insight

        except Exception as e:
            logger.error(f"LLM insight generation failed: {str(e)}")
            return self.generate_rule_based_insight(data_insights)

    def generate_rule_based_insight(self, data_insights: Dict[str, Any]) -> str:
        """Generate insights using rule-based approach when LLM fails"""
        insights = []

        if data_insights['premium_brands'] > data_insights['value_brands']:
            insights.append("The marketplace is predominantly premium-focused, indicating a high-value customer base.")
        else:
            insights.append("The marketplace shows balanced positioning between premium and value segments.")

        if data_insights['avg_discount'] > 15:
            insights.append(
                f"High average discount rate ({data_insights['avg_discount']:.1f}%) suggests aggressive promotional strategy.")
        else:
            insights.append(f"Moderate discounting strategy maintains brand value while driving sales.")

        if data_insights['optimization_opportunities'] > 0:
            insights.append(
                f"Identified {data_insights['optimization_opportunities']} pricing optimization opportunities for revenue growth.")

        insights.append(f"Focus on {data_insights['top_category']} category presents significant growth potential.")

        return " ".join(insights)

    def get_comprehensive_insights(self, df_clean: pd.DataFrame, df_competitiveness: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive insights including LLM analysis"""
        logger.info("Generating comprehensive pricing insights")

        # Generate data insights
        data_insights = self.generate_data_insights(df_clean, df_competitiveness)

        # Generate LLM insights
        llm_insight = self.generate_llm_insight(data_insights)

        # Compile comprehensive insights
        comprehensive_insights = {
            "data_insights": data_insights,
            "llm_analysis": llm_insight,
            "recommendations": [
                "Monitor premium brand pricing to maintain competitiveness",
                "Optimize discount strategies for overstocked products",
                "Develop category-specific pricing approaches",
                "Leverage data insights for dynamic pricing decisions"
            ],
            "key_metrics": {
                "marketplace_health_score": min(100, max(0, 80 + data_insights['avg_discount'] - 10)),
                "pricing_optimization_potential": min(100, data_insights['optimization_opportunities'] / data_insights[
                    'total_brands'] * 1000),
                "competitive_position": "Strong" if data_insights['premium_brands'] > data_insights[
                    'value_brands'] else "Balanced"
            }
        }

        logger.info("Comprehensive insights generated successfully")
        return comprehensive_insights


def main():
    """Test insight generation"""
    try:
        # Load data
        df_clean = pd.read_parquet("data/processed/products_cleaned.parquet")
        df_competitiveness = pd.read_parquet("data/processed/price_competitiveness_analysis.parquet")

        # Generate insights
        insight_gen = PricingInsightGenerator()
        insights = insight_gen.get_comprehensive_insights(df_clean, df_competitiveness)

        print("=== PRICING INSIGHTS ===")
        print(f"LLM Analysis: {insights['llm_analysis']}")
        print("\nKey Metrics:")
        for metric, value in insights['key_metrics'].items():
            print(f"  {metric}: {value}")
        print("\nRecommendations:")
        for rec in insights['recommendations']:
            print(f"  • {rec}")

        return insights

    except Exception as e:
        logger.error(f"Error in insight generation: {str(e)}")
        raise


if __name__ == "__main__":
    insights = main()
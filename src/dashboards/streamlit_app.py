import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pricing.price_competitiveness import PriceCompetitivenessAnalyzer
from pricing.discount_recommendation import DiscountRecommender


class PricingDashboard:
    def __init__(self):
        self.set_page_config()

    def set_page_config(self):
        st.set_page_config(
            page_title="Farfetch Pricing Analytics",
            page_icon="🛍️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def load_data(self):
        """Load processed data"""
        try:
            df_clean = pd.read_parquet("data/processed/products_cleaned.parquet")
            df_competitiveness = pd.read_parquet("data/processed/price_competitiveness_analysis.parquet")
            df_recommendations = pd.read_parquet("data/processed/discount_recommendations.parquet")
            return df_clean, df_competitiveness, df_recommendations
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None, None

    def display_overview(self, df_clean):
        """Display overview metrics"""
        st.title("🛍️ Farfetch Pricing Analytics Dashboard")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_products = len(df_clean)
            st.metric("Total Products", f"{total_products:,}")

        with col2:
            avg_price = df_clean['price_final_usd'].mean()
            st.metric("Average Price", f"${avg_price:,.2f}")

        with col3:
            discount_products = df_clean['has_discount'].sum()
            st.metric("Products on Discount", f"{discount_products:,}")

        with col4:
            avg_discount = df_clean['discount_pct'].mean()
            st.metric("Average Discount", f"{avg_discount:.1f}%")

    def display_price_analysis(self, df_clean, df_competitiveness):
        """Display price analysis visualizations"""
        st.header("📊 Price Competitiveness Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Price distribution by category
            fig1 = px.box(df_clean, x='main_category', y='price_final_usd',
                          title="Price Distribution by Category")
            fig1.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Price positioning
            positioning_counts = df_competitiveness['price_positioning'].value_counts()
            fig2 = px.pie(values=positioning_counts.values, names=positioning_counts.index,
                          title="Price Positioning Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        # Competitiveness by brand
        st.subheader("Top Brands - Competitiveness Score")
        top_brands = df_competitiveness.nlargest(10, 'price_final_usd_count')

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=top_brands['brand_clean'],
            y=top_brands['competitiveness_score'],
            name='Competitiveness Score'
        ))
        fig3.update_layout(title="Top Brands by Competitiveness Score",
                           xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)

    def display_discount_recommendations(self, df_recommendations):
        """Display discount recommendations"""
        st.header("🎯 Discount Recommendations")

        # Filter recommendations
        col1, col2 = st.columns(2)

        with col1:
            min_stock = st.slider("Minimum Stock Quantity", 0, 500, 10)
            min_discount = st.slider("Minimum Recommended Discount", 0, 70, 10)

        filtered_recs = df_recommendations[
            (df_recommendations['stock_quantity'] >= min_stock) &
            (df_recommendations['recommended_discount'] >= min_discount)
            ]

        # Display recommendations
        st.dataframe(
            filtered_recs.nlargest(20, 'recommended_discount')[
                ['brand', 'current_price', 'current_discount',
                 'recommended_discount', 'recommendation', 'stock_quantity']
            ].round(2),
            use_container_width=True
        )

        # Recommendation distribution
        rec_counts = filtered_recs['recommendation'].value_counts()
        fig = px.bar(x=rec_counts.index, y=rec_counts.values,
                     title="Recommendation Distribution")
        st.plotly_chart(fig, use_container_width=True)

    def display_llm_insights(self, df_clean, df_competitiveness):
        """Display LLM-powered insights"""
        st.header("🤖 AI-Powered Insights")

        # Generate insights based on data analysis
        total_brands = df_clean['brand_clean'].nunique()
        avg_competitiveness = df_competitiveness['competitiveness_score'].mean()
        premium_brands = len(df_competitiveness[df_competitiveness['price_positioning'] == 'Premium'])

        insights = [
            f"**Market Overview**: Analyzing {len(df_clean):,} products across {total_brands} brands",
            f"**Competitiveness**: Average competitiveness score of {avg_competitiveness:.1f}/100",
            f"**Pricing Strategy**: {premium_brands} brands positioned as premium",
            f"**Opportunity**: {len(df_competitiveness[df_competitiveness['optimization_opportunity'] != 'Maintain Current Pricing'])} brands need pricing adjustments",
            f"**Recommendation**: Focus on brands with high stock and suboptimal pricing for maximum impact"
        ]

        for insight in insights:
            st.info(insight)

    def run(self):
        """Main dashboard execution"""
        # Load data
        df_clean, df_competitiveness, df_recommendations = self.load_data()

        if df_clean is None:
            st.error("Failed to load data. Please check if the data files exist.")
            return

        # Sidebar filters
        st.sidebar.title("Filters")

        # Brand filter
        selected_brands = st.sidebar.multiselect(
            "Select Brands",
            options=sorted(df_clean['brand_clean'].unique()),
            default=sorted(df_clean['brand_clean'].unique())[:5]
        )

        # Category filter
        selected_categories = st.sidebar.multiselect(
            "Select Categories",
            options=sorted(df_clean['main_category'].unique()),
            default=sorted(df_clean['main_category'].unique())[:3]
        )

        # Price range filter
        price_range = st.sidebar.slider(
            "Price Range (USD)",
            float(df_clean['price_final_usd'].min()),
            float(df_clean['price_final_usd'].max()),
            (0.0, 1000.0)
        )

        # Apply filters
        df_filtered = df_clean[
            (df_clean['brand_clean'].isin(selected_brands)) &
            (df_clean['main_category'].isin(selected_categories)) &
            (df_clean['price_final_usd'].between(price_range[0], price_range[1]))
            ]

        # Display dashboard sections
        self.display_overview(df_filtered)
        self.display_price_analysis(df_filtered, df_competitiveness)
        self.display_discount_recommendations(df_recommendations)
        self.display_llm_insights(df_filtered, df_competitiveness)


def main():
    dashboard = PricingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import plotly.io as pio
from datetime import datetime
import os
from pathlib import Path
import logging
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PricingReportGenerator:
    def __init__(self):
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
        self.plots_dir = self.report_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_plots(self, df_clean: pd.DataFrame, df_competitiveness: pd.DataFrame, df_recommendations: pd.DataFrame):
        """Create visualization plots for the report"""
        plots = {}

        try:
            # 1. Price distribution by category
            plt.figure(figsize=(12, 8))
            top_categories = df_clean['main_category'].value_counts().head(10).index
            df_top_cats = df_clean[df_clean['main_category'].isin(top_categories)]

            sns.boxplot(data=df_top_cats, x='main_category', y='price_final_usd')
            plt.xticks(rotation=45)
            plt.title('Price Distribution by Category')
            plt.tight_layout()
            plots['price_distribution'] = self.plots_dir / "price_distribution.png"
            plt.savefig(plots['price_distribution'], dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Competitiveness score distribution
            plt.figure(figsize=(10, 6))
            plt.hist(df_competitiveness['competitiveness_score'], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Competitiveness Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Competitiveness Scores')
            plt.grid(True, alpha=0.3)
            plots['competitiveness_dist'] = self.plots_dir / "competitiveness_dist.png"
            plt.savefig(plots['competitiveness_dist'], dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Price positioning
            plt.figure(figsize=(8, 8))
            positioning_counts = df_competitiveness['price_positioning'].value_counts()
            plt.pie(positioning_counts.values, labels=positioning_counts.index, autopct='%1.1f%%')
            plt.title('Price Positioning Distribution')
            plots['positioning_pie'] = self.plots_dir / "positioning_pie.png"
            plt.savefig(plots['positioning_pie'], dpi=300, bbox_inches='tight')
            plt.close()

            # 4. Top recommended discounts
            plt.figure(figsize=(12, 6))
            top_recs = df_recommendations.nlargest(15, 'recommended_discount')
            sns.barplot(data=top_recs, x='brand', y='recommended_discount')
            plt.xticks(rotation=45)
            plt.title('Top Recommended Discounts by Brand')
            plt.tight_layout()
            plots['top_discounts'] = self.plots_dir / "top_discounts.png"
            plt.savefig(plots['top_discounts'], dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("All plots created successfully")
            return plots

        except Exception as e:
            logger.error(f"Error creating plots: {str(e)}")
            raise

    def generate_pdf_report(self, df_clean: pd.DataFrame, df_competitiveness: pd.DataFrame,
                            df_recommendations: pd.DataFrame):
        """Generate comprehensive PDF report"""
        logger.info("Generating PDF report")

        # Create plots
        plots = self.create_plots(df_clean, df_competitiveness, df_recommendations)

        # Initialize PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Add cover page
        pdf.add_page()
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 40, 'Farfetch Pricing Analytics Report', 0, 1, 'C')
        pdf.set_font('Arial', 'I', 14)
        pdf.cell(0, 20, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        pdf.ln(20)

        # Executive Summary
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Executive Summary', 0, 1)
        pdf.set_font('Arial', '', 12)

        summary_stats = [
            f"Total Products Analyzed: {len(df_clean):,}",
            f"Total Brands: {df_clean['brand_clean'].nunique():,}",
            f"Average Price: ${df_clean['price_final_usd'].mean():.2f}",
            f"Average Discount: {df_clean['discount_pct'].mean():.1f}%",
            f"Products on Discount: {df_clean['has_discount'].sum():,}",
            f"Average Competitiveness Score: {df_competitiveness['competitiveness_score'].mean():.1f}/100",
            f"Optimization Opportunities: {len(df_competitiveness[df_competitiveness['optimization_opportunity'] != 'Maintain Current Pricing']):,}"
        ]

        for stat in summary_stats:
            pdf.cell(0, 8, stat, 0, 1)

        # Key Insights
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Key Insights', 0, 1)
        pdf.set_font('Arial', '', 12)

        insights = [
            "• Premium positioning brands show highest price premiums",
            "• Significant discount optimization opportunities identified",
            "• Stock levels correlate with pricing competitiveness",
            "• Category-level pricing strategies vary significantly"
        ]

        for insight in insights:
            pdf.multi_cell(0, 8, insight)

        # Add plots to PDF
        for plot_name, plot_path in plots.items():
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)

            titles = {
                'price_distribution': 'Price Distribution by Category',
                'competitiveness_dist': 'Competitiveness Score Distribution',
                'positioning_pie': 'Price Positioning Analysis',
                'top_discounts': 'Top Recommended Discounts'
            }

            pdf.cell(0, 10, titles.get(plot_name, 'Chart'), 0, 1)
            pdf.image(str(plot_path), x=10, y=30, w=190)

        # Recommendations
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Strategic Recommendations', 0, 1)
        pdf.set_font('Arial', '', 12)

        recommendations = [
            "1. Implement dynamic pricing for overstocked premium products",
            "2. Review pricing strategy for brands with low competitiveness scores",
            "3. Optimize discount levels based on stock turnover rates",
            "4. Monitor competitor pricing for key luxury categories",
            "5. Develop category-specific pricing strategies"
        ]

        for rec in recommendations:
            pdf.multi_cell(0, 8, rec)

        # Save PDF
        report_path = self.report_dir / f"pricing_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        pdf.output(str(report_path))

        logger.info(f"PDF report generated: {report_path}")
        return report_path


def main():
    """Generate comprehensive pricing report"""
    try:
        # Load data
        df_clean = pd.read_parquet("data/processed/products_cleaned.parquet")
        df_competitiveness = pd.read_parquet("data/processed/price_competitiveness_analysis.parquet")
        df_recommendations = pd.read_parquet("data/processed/discount_recommendations.parquet")

        # Generate report
        generator = PricingReportGenerator()
        report_path = generator.generate_pdf_report(df_clean, df_competitiveness, df_recommendations)

        print(f"Report generated successfully: {report_path}")
        return report_path

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise


if __name__ == "__main__":
    main()
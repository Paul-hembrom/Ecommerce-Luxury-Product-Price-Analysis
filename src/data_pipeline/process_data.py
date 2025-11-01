import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import sys
import os

# Add src to path for proper imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Now import from utils
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataProcessingPipeline:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.data_dir = Path("data")
        self.setup_directories()

    def setup_directories(self):
        """Create all necessary directories"""
        directories = [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "external",
            self.data_dir / "interim",
            self.data_dir / "reports"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")



    def load_raw_data(self) -> pd.DataFrame:
        """Load and validate raw JSON data"""
        logger.info(f"Loading raw data from: {self.raw_data_path}")

        try:
            with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Basic validation
            required_columns = ['id', 'brand', 'price_final', 'price_full']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            logger.info(f"Successfully loaded {len(df)} products with {len(df.columns)} columns")
            logger.info(f"Columns: {df.columns.tolist()}")

            return df

        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise

    def run_cleaning_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run data cleaning pipeline"""
        logger.info("Starting data cleaning pipeline")

        cleaner = DataCleaner()
        df_clean = cleaner.clean_product_data(df)

        # Save cleaned data
        output_path = self.data_dir / "processed" / "products_cleaned.parquet"
        df_clean.to_parquet(output_path, index=False)
        logger.info(f"Cleaned data saved to: {output_path}")

        # Generate cleaning report
        self.generate_cleaning_report(df, df_clean)

        return df_clean

    def generate_cleaning_report(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame):
        """Generate data cleaning report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'raw_data_stats': {
                'total_products': len(df_raw),
                'total_columns': len(df_raw.columns),
                'missing_values': df_raw.isnull().sum().to_dict(),
                'data_types': df_raw.dtypes.astype(str).to_dict()
            },
            'cleaning_operations': {
                'products_after_cleaning': len(df_clean),
                'columns_after_cleaning': len(df_clean.columns),
                'price_conversion': f"{len(df_clean) - df_clean['price_final_usd'].isna().sum()}/{len(df_clean)} products converted",
                'brands_cleaned': df_clean['brand_clean'].nunique(),
                'categories_extracted': df_clean['main_category'].nunique()
            },
            'new_features_created': [
                'price_final_usd', 'price_full_usd', 'discount_pct', 'main_category',
                'sub_category', 'brand_clean', 'has_discount', 'price_segment'
            ]
        }

        # Save report
        report_path = self.data_dir / "interim" / "cleaning_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Cleaning report saved to: {report_path}")

    def run_competitiveness_analysis(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Run price competitiveness analysis"""
        logger.info("Starting price competitiveness analysis")

        analyzer = PriceCompetitivenessAnalyzer()
        df_competitiveness = analyzer.calculate_price_positioning(df_clean)
        df_competitiveness = analyzer.identify_opportunities(df_competitiveness)

        # Save competitiveness analysis
        output_path = self.data_dir / "processed" / "price_competitiveness_analysis.parquet"
        df_competitiveness.to_parquet(output_path, index=False)
        logger.info(f"Competitiveness analysis saved to: {output_path}")

        # Generate competitiveness insights
        self.generate_competitiveness_insights(df_competitiveness)

        return df_competitiveness

    def generate_competitiveness_insights(self, df_competitiveness: pd.DataFrame):
        """Generate competitiveness insights report"""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'summary_stats': {
                'total_brand_category_combinations': len(df_competitiveness),
                'average_competitiveness_score': df_competitiveness['competitiveness_score'].mean(),
                'premium_positioned': len(df_competitiveness[df_competitiveness['price_positioning'] == 'Premium']),
                'value_positioned': len(df_competitiveness[df_competitiveness['price_positioning'] == 'Value']),
                'competitive_positioned': len(
                    df_competitiveness[df_competitiveness['price_positioning'] == 'Competitive'])
            },
            'optimization_opportunities': {
                'price_reduction_opportunities': len(
                    df_competitiveness[df_competitiveness['optimization_opportunity'] == 'Consider Price Reduction']),
                'price_increase_opportunities': len(
                    df_competitiveness[df_competitiveness['optimization_opportunity'] == 'Potential Price Increase']),
                'maintain_pricing': len(
                    df_competitiveness[df_competitiveness['optimization_opportunity'] == 'Maintain Current Pricing'])
            },
            'top_performers': {
                'most_competitive_brands': df_competitiveness.nlargest(5, 'competitiveness_score')[
                    ['brand_clean', 'competitiveness_score']].to_dict('records'),
                'least_competitive_brands': df_competitiveness.nsmallest(5, 'competitiveness_score')[
                    ['brand_clean', 'competitiveness_score']].to_dict('records')
            }
        }

        # Save insights
        insights_path = self.data_dir / "interim" / "competitiveness_insights.json"
        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2)

        logger.info(f"Competitiveness insights saved to: {insights_path}")

    def run_discount_recommendations(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Run discount recommendation engine"""
        logger.info("Starting discount recommendation analysis")

        # Filter valid products for discount analysis
        df_valid = df_clean[
            (df_clean['price_final_usd'] > 0) &
            (df_clean['price_final_usd'] < 10000) &  # Remove extreme outliers
            (df_clean['discount_pct'] >= 0) &
            (df_clean['discount_pct'] <= 80)  # Reasonable discount range
            ].copy()

        recommender = DiscountRecommender()
        df_recommendations = recommender.recommend_discounts(df_valid)

        # Save recommendations
        output_path = self.data_dir / "processed" / "discount_recommendations.parquet"
        df_recommendations.to_parquet(output_path, index=False)
        logger.info(f"Discount recommendations saved to: {output_path}")

        # Generate discount insights
        self.generate_discount_insights(df_recommendations)

        return df_recommendations

    def generate_discount_insights(self, df_recommendations: pd.DataFrame):
        """Generate discount insights report"""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'summary_stats': {
                'total_recommendations': len(df_recommendations),
                'increase_discount_recommendations': len(
                    df_recommendations[df_recommendations['recommendation'] == 'Increase Discount']),
                'decrease_discount_recommendations': len(
                    df_recommendations[df_recommendations['recommendation'] == 'Decrease Discount']),
                'maintain_discount_recommendations': len(
                    df_recommendations[df_recommendations['recommendation'] == 'Maintain Current Discount']),
                'average_recommended_discount': df_recommendations['recommended_discount'].mean(),
                'average_current_discount': df_recommendations['current_discount'].mean()
            },
            'impact_analysis': {
                'potential_price_changes': (
                            df_recommendations['recommended_price'] - df_recommendations['current_price']).sum(),
                'average_price_change': (
                            df_recommendations['recommended_price'] - df_recommendations['current_price']).mean(),
                'max_recommended_discount': df_recommendations['recommended_discount'].max(),
                'min_recommended_discount': df_recommendations['recommended_discount'].min()
            }
        }

        # Save insights
        insights_path = self.data_dir / "interim" / "discount_insights.json"
        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2)

        logger.info(f"Discount insights saved to: {insights_path}")

    def run_elasticity_analysis(self, df_clean: pd.DataFrame):
        """Run price elasticity analysis"""
        logger.info("Starting price elasticity analysis")

        try:
            elasticity_model = PriceElasticityModel()
            results = elasticity_model.train_elasticity_model(df_clean)

            # Get optimization recommendations
            recommendations = elasticity_model.get_optimization_recommendations(df_clean, top_n=50)

            # Save elasticity results
            model_path = self.data_dir / "processed" / "elasticity_results.parquet"
            recommendations.to_parquet(model_path, index=False)

            # Save model as .pkl using joblib
            model_save_path = "models/price_elasticity_model.pkl"
            elasticity_model.save_model(model_save_path)

            logger.info(f"Elasticity analysis completed. Recommendations saved to: {model_path}")
            logger.info(f"Elasticity model saved to: {model_save_path}")

            return elasticity_model, recommendations

        except Exception as e:
            logger.error(f"Elasticity analysis failed: {str(e)}")
            # Return empty results but don't break the pipeline
            return None, pd.DataFrame()

    def generate_master_dataset(self, df_clean: pd.DataFrame, df_competitiveness: pd.DataFrame,
                                df_recommendations: pd.DataFrame):
        """Generate master dataset with all features"""
        logger.info("Creating master dataset")

        # Merge all datasets
        master_df = df_clean.merge(
            df_competitiveness[['brand_clean', 'main_category', 'price_positioning',
                                'competitiveness_score', 'optimization_opportunity']],
            on=['brand_clean', 'main_category'],
            how='left'
        )

        master_df = master_df.merge(
            df_recommendations[['product_id', 'recommended_discount', 'recommendation', 'recommended_price']],
            left_on='id',
            right_on='product_id',
            how='left'
        )

        # Add derived features for master dataset
        master_df['price_gap_to_recommended'] = master_df['price_final_usd'] - master_df['recommended_price']
        master_df['discount_gap'] = master_df['recommended_discount'] - master_df['discount_pct']

        # Save master dataset
        output_path = self.data_dir / "processed" / "master_pricing_dataset.parquet"
        master_df.to_parquet(output_path, index=False)
        logger.info(f"Master dataset saved to: {output_path}")

        return master_df

    def run_complete_pipeline(self):
        """Run complete data processing pipeline"""
        logger.info("Starting complete data processing pipeline")

        try:
            # Step 1: Load raw data
            df_raw = self.load_raw_data()

            # Step 2: Data cleaning
            df_clean = self.run_cleaning_pipeline(df_raw)

            # Step 3: Competitiveness analysis
            df_competitiveness = self.run_competitiveness_analysis(df_clean)

            # Step 4: Discount recommendations
            df_recommendations = self.run_discount_recommendations(df_clean)

            # Step 5: Elasticity analysis
            elasticity_model, elasticity_recommendations = self.run_elasticity_analysis(df_clean)

            # Step 6: Create master dataset
            master_df = self.generate_master_dataset(df_clean, df_competitiveness, df_recommendations)

            # Step 7: Generate final pipeline report
            self.generate_pipeline_report(df_raw, df_clean, df_competitiveness,
                                          df_recommendations, elasticity_recommendations)

            logger.info("Data processing pipeline completed successfully!")

            return {
                'raw_data': df_raw,
                'cleaned_data': df_clean,
                'competitiveness_analysis': df_competitiveness,
                'discount_recommendations': df_recommendations,
                'elasticity_recommendations': elasticity_recommendations,
                'master_dataset': master_df
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def generate_pipeline_report(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                                 df_competitiveness: pd.DataFrame, df_recommendations: pd.DataFrame,
                                 elasticity_recommendations: pd.DataFrame):
        """Generate comprehensive pipeline report"""
        report = {
            'pipeline_execution': {
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'datasets_generated': [
                    'products_cleaned.parquet',
                    'price_competitiveness_analysis.parquet',
                    'discount_recommendations.parquet',
                    'elasticity_results.parquet',
                    'master_pricing_dataset.parquet'
                ]
            },
            'data_quality_metrics': {
                'initial_products': len(df_raw),
                'cleaned_products': len(df_clean),
                'data_retention_rate': f"{(len(df_clean) / len(df_raw)) * 100:.1f}%",
                'brands_analyzed': df_clean['brand_clean'].nunique(),
                'categories_analyzed': df_clean['main_category'].nunique()
            },
            'analytics_insights': {
                'competitiveness_analysis_combinations': len(df_competitiveness),
                'discount_recommendations_generated': len(df_recommendations),
                'elasticity_optimization_opportunities': len(elasticity_recommendations),
                'average_competitiveness_score': df_competitiveness['competitiveness_score'].mean(),
                'premium_products_percentage': f"{(len(df_clean[df_clean['price_segment'] == 'Premium']) / len(df_clean)) * 100:.1f}%"
            },
            'business_impact': {
                'products_with_optimization_opportunities': len(
                    df_competitiveness[df_competitiveness['optimization_opportunity'] != 'Maintain Current Pricing']),
                'potential_revenue_impact_elasticity': elasticity_recommendations[
                    'predicted_revenue_change_pct'].sum() if len(elasticity_recommendations) > 0 else 0,
                'high_impact_recommendations': len(
                    elasticity_recommendations[elasticity_recommendations['predicted_revenue_change_pct'] > 5]) if len(
                    elasticity_recommendations) > 0 else 0
            }
        }

        # Save comprehensive report
        report_path = self.data_dir / "reports" / "pipeline_execution_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Pipeline execution report saved to: {report_path}")

        # Also save as formatted text report
        self.save_text_report(report, report_path.with_suffix('.txt'))

    def save_text_report(self, report: dict, filepath: Path):
        """Save report as formatted text file"""
        with open(filepath, 'w') as f:
            f.write("FARFETCH PRICING ANALYTICS - DATA PROCESSING PIPELINE REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Execution Time: {report['pipeline_execution']['timestamp']}\n")
            f.write(f"Status: {report['pipeline_execution']['status'].upper()}\n\n")

            f.write("DATA QUALITY METRICS:\n")
            f.write("-" * 40 + "\n")
            for key, value in report['data_quality_metrics'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")

            f.write("\nANALYTICS INSIGHTS:\n")
            f.write("-" * 40 + "\n")
            for key, value in report['analytics_insights'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")

            f.write("\nBUSINESS IMPACT:\n")
            f.write("-" * 40 + "\n")
            for key, value in report['business_impact'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")

            f.write(f"\nGenerated Datasets ({len(report['pipeline_execution']['datasets_generated'])}):\n")
            for dataset in report['pipeline_execution']['datasets_generated']:
                f.write(f"  - {dataset}\n")


def main():
    """Main function to run the data processing pipeline"""
    raw_data_path = "D:/Farfetch/farfetch_men_allclothing_products1.json"

    pipeline = DataProcessingPipeline(raw_data_path)
    results = pipeline.run_complete_pipeline()

    print("\n" + "=" * 70)
    print("DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print(f"✅ Raw data loaded: {len(results['raw_data']):,} products")
    print(f"✅ Data cleaned: {len(results['cleaned_data']):,} products")
    print(f"✅ Competitiveness analysis: {len(results['competitiveness_analysis']):,} brand-category combinations")
    print(f"✅ Discount recommendations: {len(results['discount_recommendations']):,} products")
    print(f"✅ Elasticity recommendations: {len(results['elasticity_recommendations']):,} optimizations")
    print(f"✅ Master dataset created: {len(results['master_dataset']):,} products with all features")

    print(f"\n📊 Generated files in 'data/processed/':")
    print("  - products_cleaned.parquet")
    print("  - price_competitiveness_analysis.parquet")
    print("  - discount_recommendations.parquet")
    print("  - elasticity_results.parquet")
    print("  - master_pricing_dataset.parquet")

    print(f"\n📈 Interim files in 'data/interim/':")
    print("  - cleaning_report.json")
    print("  - competitiveness_insights.json")
    print("  - discount_insights.json")

    print(f"\n📋 Report in 'data/reports/':")
    print("  - pipeline_execution_report.json")
    print("  - pipeline_execution_report.txt")


if __name__ == "__main__":
    main()
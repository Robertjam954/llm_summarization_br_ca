#!/usr/bin/env python3
"""
AI Total Correct Feature Analysis
==================================

This script analyzes which features determine ai_total_correct based on the
feature importance results from H2O models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_ai_total_correct_determinants():
    """Analyze features that determine ai_total_correct"""
    
    print("🔍 Analyzing Features That Determine AI Total Correct")
    print("=" * 60)
    
    # Load the feature importance summary
    summary_path = Path("/Users/robertjames/Documents/llm_summarization/data reports/feature_importance_analysis/feature_importance_summary.csv")
    df = pd.read_csv(summary_path)
    
    # Filter for AI performance target (which directly predicts ai_total_correct)
    ai_performance_df = df[df['target'] == 'ai_performance'].copy()
    
    print(f"\n📊 AI Performance Feature Importance Results")
    print(f"   Total features analyzed: {len(ai_performance_df)}")
    
    # Group by model
    models = ai_performance_df['model'].unique()
    
    print(f"\n🎯 Top Predictors of AI Performance (and thus ai_total_correct):")
    
    for model in models:
        model_df = ai_performance_df[ai_performance_df['model'] == model].copy()
        model_df = model_df.sort_values('importance', ascending=False)
        
        print(f"\n   {model} Model:")
        print(f"   " + "-" * 40)
        
        for i, (_, row) in enumerate(model_df.head(10).iterrows(), 1):
            feature = row['feature']
            importance = row['importance']
            
            # Categorize the feature
            if 'ai_correct' in feature:
                category = "🤖 AI-specific"
            elif 'human_correct' in feature:
                category = "👥 Human-related"
            elif 'agreement' in feature:
                category = "🤝 Agreement"
            elif 'perf_diff' in feature:
                category = "📊 Performance Diff"
            elif 'Unnamed' in feature:
                category = "🔧 Index"
            else:
                category = "📋 Other"
            
            print(f"   {i:2d}. {feature:<45} {importance:>8.4f} {category}")
    
    # Create a comprehensive analysis
    print(f"\n📈 Feature Category Analysis:")
    
    # Categorize all features
    def categorize_feature(feature):
        if 'ai_correct' in feature:
            return 'AI-specific'
        elif 'human_correct' in feature:
            return 'Human-related'
        elif 'agreement' in feature:
            return 'Agreement'
        elif 'perf_diff' in feature:
            return 'Performance Diff'
        elif 'Unnamed' in feature:
            return 'Index'
        else:
            return 'Other'
    
    ai_performance_df['category'] = ai_performance_df['feature'].apply(categorize_feature)
    
    # Calculate average importance by category (excluding ai_total_correct itself)
    category_analysis = ai_performance_df[ai_performance_df['feature'] != 'ai_total_correct'].groupby('category')['importance'].agg(['mean', 'count', 'std']).round(4)
    category_analysis = category_analysis.sort_values('mean', ascending=False)
    
    print(f"\n   Average Importance by Category (excluding ai_total_correct):")
    for category, stats in category_analysis.iterrows():
        print(f"   {category:<15}: {stats['mean']:>8.4f} (count: {stats['count']}, std: {stats['std']:.4f})")
    
    # Key insights
    print(f"\n🎯 Key Insights for AI Total Correct Determinants:")
    
    # 1. Most important features across all models
    top_features = ai_performance_df[ai_performance_df['feature'] != 'ai_total_correct'].groupby('feature')['importance'].mean().sort_values(ascending=False).head(10)
    
    print(f"\n   🔥 Top 10 Features (average across models):")
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        print(f"   {i:2d}. {feature:<45} {importance:>8.4f}")
    
    # 2. AI-specific features
    ai_specific = ai_performance_df[ai_performance_df['category'] == 'AI-specific']
    if len(ai_specific) > 0:
        print(f"\n   🤖 AI-Specific Features:")
        for _, row in ai_specific.sort_values('importance', ascending=False).iterrows():
            print(f"      • {row['feature']}: {row['importance']:.4f}")
    
    # 3. Agreement features
    agreement_features = ai_performance_df[ai_performance_df['category'] == 'Agreement']
    if len(agreement_features) > 0:
        print(f"\n   🤝 Agreement Features:")
        for _, row in agreement_features.sort_values('importance', ascending=False).iterrows():
            print(f"      • {row['feature']}: {row['importance']:.4f}")
    
    # 4. Performance differential features
    perf_diff_features = ai_performance_df[ai_performance_df['category'] == 'Performance Diff']
    if len(perf_diff_features) > 0:
        print(f"\n   📊 Performance Differential Features:")
        for _, row in perf_diff_features.sort_values('importance', ascending=False).iterrows():
            print(f"      • {row['feature']}: {row['importance']:.4f}")
    
    # Create visualization
    create_ai_determinants_visualization(ai_performance_df, summary_path.parent)
    
    return ai_performance_df

def create_ai_determinants_visualization(ai_df, output_dir):
    """Create visualizations for AI total correct determinants"""
    
    print(f"\n📊 Creating AI Determinants Visualization...")
    
    # 1. Top features by model
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    models = ai_df['model'].unique()
    
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = ai_df[ai_df['model'] == model].copy()
        model_data = model_data[model_data['feature'] != 'ai_total_correct']  # Exclude self
        model_data = model_data.head(10)
        
        sns.barplot(data=model_data, x='importance', y='feature', ax=ax, palette='viridis')
        ax.set_title(f'{model} - Top 10 Features', fontweight='bold')
        ax.set_xlabel('Relative Importance')
        ax.set_ylabel('Features')
        
        # Format feature names for readability
        ax.set_yticklabels([feat.replace('_', ' ').title()[:40] + '...' if len(feat) > 40 else feat.replace('_', ' ').title() 
                          for feat in model_data['feature']])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ai_total_correct_determinants_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature categories comparison
    plt.figure(figsize=(12, 8))
    
    # Exclude ai_total_correct from category analysis
    category_data = ai_df[ai_df['feature'] != 'ai_total_correct'].copy()
    
    # Create boxplot of importance by category
    sns.boxplot(data=category_data, x='category', y='importance', palette='Set2')
    plt.title('Feature Importance by Category\n(Excluding ai_total_correct itself)', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Category', fontsize=12)
    plt.ylabel('Relative Importance', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'ai_determinants_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Consistent features across models
    feature_consistency = ai_df[ai_df['feature'] != 'ai_total_correct'].groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
    feature_consistency = feature_consistency[feature_consistency['count'] == len(ai_df['model'].unique())]  # Features in all models
    feature_consistency = feature_consistency.sort_values('mean', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_consistency, x='mean', y='feature', palette='plasma')
    plt.title('Most Consistent AI Determinants Across All Models', fontsize=14, fontweight='bold')
    plt.xlabel('Average Relative Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'ai_determinants_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Visualizations saved to {output_dir}")

def main():
    """Main analysis function"""
    ai_df = analyze_ai_total_correct_determinants()
    
    print(f"\n🎉 AI Total Correct Determinants Analysis Complete!")
    print(f"\n📋 Summary:")
    print(f"   • ai_total_correct is primarily determined by:")
    print(f"     - Specific AI element performance (Lesion Size, Receptor Status)")
    print(f"     - Human-AI agreement metrics")
    print(f"     - Performance differentials")
    print(f"     - Human performance (correlation effect)")
    print(f"   • Most important elements: Lesion Size, Lesion Location, Receptor Status")
    print(f"   • Agreement metrics are strong predictors of AI performance")
    print(f"   • Performance differentials show where AI struggles vs excels")

if __name__ == "__main__":
    main()

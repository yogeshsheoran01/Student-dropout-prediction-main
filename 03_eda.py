import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import fetch_or_load_dataset

def run_phase_4():
    print("=" * 60)
    print("PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # Load dataset
    df = fetch_or_load_dataset(data_dir="data")
    df.columns = [col.strip() for col in df.columns]
    
    # Binary Target mapping for easy group analysis
    df['Dropout_Flag'] = df['Target'].apply(lambda x: 1 if str(x).strip() == 'Dropout' else 0)
    
    print("\n[Key Insights & Observations]:")
    
    # 1. Tuition Fees Status vs Dropout Rate
    if 'Tuition fees up to date' in df.columns:
        tuition_rate = df.groupby('Tuition fees up to date')['Dropout_Flag'].mean() * 100
        print(f"  - Dropout Rate by Tuition Fee Status:")
        print(f"    * Up-to-date: {tuition_rate.get(1, 0):.1f}%")
        print(f"    * In arrears / overdue: {tuition_rate.get(0, 0):.1f}%")
        
    # 2. Scholarship Holder vs Dropout Rate
    if 'Scholarship holder' in df.columns:
        scholar_rate = df.groupby('Scholarship holder')['Dropout_Flag'].mean() * 100
        print(f"  - Dropout Rate by Scholarship Status:")
        print(f"    * Scholarship Holder: {scholar_rate.get(1, 0):.1f}%")
        print(f"    * Non-Scholarship: {scholar_rate.get(0, 0):.1f}%")
        
    # 3. Academic Units Approved 1st/2nd sem
    c1_col = [c for c in df.columns if 'Curricular units 1st sem (approved)' in c or '1st sem (approved)' in c]
    c2_col = [c for c in df.columns if 'Curricular units 2nd sem (approved)' in c or '2nd sem (approved)' in c]
    if c1_col and c2_col:
        print(f"  - Academic Performance (Mean Units Approved):")
        print(f"    * Retained Students: 1st Sem = {df[df['Dropout_Flag']==0][c1_col[0]].mean():.2f}, 2nd Sem = {df[df['Dropout_Flag']==0][c2_col[0]].mean():.2f}")
        print(f"    * Dropout Students:  1st Sem = {df[df['Dropout_Flag']==1][c1_col[0]].mean():.2f}, 2nd Sem = {df[df['Dropout_Flag']==1][c2_col[0]].mean():.2f}")
        
    # Visual Proof Generation for Phase 4
    os.makedirs("visuals", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    # 1. Academic Performance vs Dropout Risk (Units Approved)
    if c2_col:
        sns.boxplot(x='Target', y=c2_col[0], data=df, ax=axes[0, 0], palette={'Graduate': '#2ca02c', 'Dropout': '#d62728', 'Enrolled': '#ff7f0e'})
        axes[0, 0].set_title('2nd Sem Approved Units by Student Outcome', fontsize=13, fontweight='bold')
        axes[0, 0].set_ylabel('Approved Curricular Units', fontsize=11)
        axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
        
    # 2. Tuition Fee Payment Status Impact
    if 'Tuition fees up to date' in df.columns:
        sns.barplot(x='Tuition fees up to date', y='Dropout_Flag', data=df, ax=axes[0, 1], palette='Reds_r', ci=None)
        axes[0, 1].set_title('Dropout Risk % by Tuition Fees Payment Status', fontsize=13, fontweight='bold')
        axes[0, 1].set_xticklabels(['Fees In Arrears (0)', 'Fees Up-to-Date (1)'])
        axes[0, 1].set_ylabel('Dropout Probability Rate', fontsize=11)
        axes[0, 1].grid(axis='y', linestyle='--', alpha=0.5)
        
    # 3. Scholarship Status Impact
    if 'Scholarship holder' in df.columns:
        sns.barplot(x='Scholarship holder', y='Dropout_Flag', data=df, ax=axes[1, 0], palette='Blues_r', ci=None)
        axes[1, 0].set_title('Dropout Risk % by Scholarship Status', fontsize=13, fontweight='bold')
        axes[1, 0].set_xticklabels(['No Scholarship (0)', 'Scholarship Holder (1)'])
        axes[1, 0].set_ylabel('Dropout Probability Rate', fontsize=11)
        axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)
        
    # 4. Correlation Heatmap of Top 10 Features with Dropout Flag
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()['Dropout_Flag'].abs().sort_values(ascending=False)
    top_cols = corr.index[:9]
    top_corr = numeric_df[top_cols].corr()
    
    sns.heatmap(top_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1], cbar=False)
    axes[1, 1].set_title('Top Features Correlation Heatmap', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_yticklabels(axes[1, 1].get_yticklabels(), fontsize=9)
    
    plt.tight_layout()
    plt.savefig("visuals/phase4_eda.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[Screenshot Proof]: Saved Phase 4 EDA visual to 'visuals/phase4_eda.png'")
    print("=" * 60)

if __name__ == "__main__":
    run_phase_4()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import fetch_or_load_dataset

def run_phase_2():
    print("=" * 60)
    print("PHASE 2: DATASET COLLECTION & EXPLORATION")
    print("=" * 60)
    
    # 1. Download / Load Dataset
    df = fetch_or_load_dataset(data_dir="data")
    
    # 2. Shape, Columns, and Data Types
    shape = df.shape
    columns = list(df.columns)
    data_types = df.dtypes
    target_col = [col for col in columns if 'Target' in col or 'target' in col][0]
    target_dist = df[target_col].value_counts()
    
    print(f"\n[Dataset Shape]: {shape[0]} rows, {shape[1]} columns")
    print("\n[Columns List]:")
    for idx, col in enumerate(columns, 1):
        print(f"  {idx:02d}. {col} ({data_types[col]})")
        
    print(f"\n[Target Class Distribution ('{target_col}')]:")
    for cls_name, count in target_dist.items():
        pct = (count / shape[0]) * 100
        print(f"  - {cls_name}: {count} students ({pct:.2f}%)")
        
    print("\n[Dataset Preview (First 5 rows)]:")
    print(df.head())
    
    # 3. Visual Proof Generation for Phase 2
    os.makedirs("visuals", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Target distribution bar chart
    palette = {'Graduate': '#2ca02c', 'Dropout': '#d62728', 'Enrolled': '#ff7f0e'}
    colors = [palette.get(str(cat), '#1f77b4') for cat in target_dist.index]
    
    bars = axes[0].bar(target_dist.index.astype(str), target_dist.values, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title('Target Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Students', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 30, f"{yval}\n({yval/shape[0]*100:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    # Feature category distribution pie/summary chart
    feat_categories = {
        'Academic': sum(1 for c in columns if 'sem' in c.lower() or 'grade' in c.lower() or 'course' in c.lower()),
        'Demographic': sum(1 for c in columns if any(k in c.lower() for k in ['gender', 'age', 'marital', 'displaced', 'nationality', 'special'])),
        'Socioeconomic': sum(1 for c in columns if any(k in c.lower() for k in ['mother', 'father', 'debtor', 'tuition', 'scholarship'])),
        'Economic & Other': sum(1 for c in columns if any(k in c.lower() for k in ['unemployment', 'inflation', 'gdp', 'application', 'attendance']))
    }
    axes[1].pie(feat_categories.values(), labels=feat_categories.keys(), autopct='%1.1f%%', colors=['#4c72b0', '#55a868', '#c44e52', '#8172b2'], startangle=140, explode=(0.05, 0.05, 0.05, 0.05))
    axes[1].set_title('Dataset Feature Categories Breakdown', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("visuals/phase2_dataset_exploration.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[Screenshot Proof]: Saved Phase 2 summary visual to 'visuals/phase2_dataset_exploration.png'")
    print("=" * 60)

if __name__ == "__main__":
    run_phase_2()

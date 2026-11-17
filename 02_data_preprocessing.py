import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import fetch_or_load_dataset
from src.preprocessing import preprocess_data

def run_phase_3():
    print("=" * 60)
    print("PHASE 3: DATA CLEANING & PREPROCESSING")
    print("=" * 60)
    
    # 1. Load Raw Dataset
    df = fetch_or_load_dataset(data_dir="data")
    
    # 2. Preprocess Data
    prep_res = preprocess_data(df, target_col='Target', save_dir='models')
    
    X = prep_res['X']
    y = prep_res['y']
    X_train_scaled = prep_res['X_train_scaled']
    X_test_scaled = prep_res['X_test_scaled']
    
    print("\n[Missing Value Analysis]:")
    missing_sum = prep_res['missing_summary']
    if missing_sum.sum() == 0:
        print("  - Zero missing values detected across all features.")
    else:
        print(missing_sum[missing_sum > 0])
        
    print("\n[Target Binary Encoding Results]:")
    print("  - Binary Target Mapping: Dropout -> 1 (At Risk), Graduate/Enrolled -> 0 (Retained)")
    print(f"  - Target Class Counts:\n{y.value_counts()}")
    
    print(f"\n[Feature Matrix Shape]: X = {X.shape}, y = {y.shape}")
    print(f"[Train-Test Split (80/20)]: Train = {X_train_scaled.shape[0]} samples, Test = {X_test_scaled.shape[0]} samples")
    print(f"[Standardization]: Features scaled to zero mean & unit variance using StandardScaler.")
    
    # Export processed dataframe
    processed_df = pd.concat([X, y], axis=1)
    processed_path = os.path.join("data", "processed_student_data.csv")
    processed_df.to_csv(processed_path, index=False)
    print(f"\n[Exported Dataset]: Saved cleaned feature dataset to '{processed_path}'")
    
    # Visual Proof Generation for Phase 3
    os.makedirs("visuals", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Binary Target Distribution
    b_counts = y.value_counts()
    bars = axes[0].bar(['Retained (0)', 'Dropout (1)'], [b_counts[0], b_counts[1]], color=['#2ca02c', '#d62728'], edgecolor='black', alpha=0.85)
    axes[0].set_title('Binary Target Distribution (Dropout vs Retained)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Student Count', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 30, f"{yval}\n({yval/len(y)*100:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    # 2. Scaled Feature Distributions Preview (Sample 4 Key Features)
    sample_cols = [c for c in X.columns if 'sem' in c.lower() or 'grade' in c.lower() or 'tuition' in c.lower()][:4]
    sample_df = pd.DataFrame(X_train_scaled, columns=X.columns)[sample_cols]
    sns.kdeplot(data=sample_df, ax=axes[1], palette='tab10', linewidth=2)
    axes[1].set_title('Standardized Feature Distributions (Mean=0, Std=1)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Scaled Value (Z-Score)', fontsize=12)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("visuals/phase3_data_preprocessing.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Screenshot Proof]: Saved Phase 3 summary visual to 'visuals/phase3_data_preprocessing.png'")
    print("=" * 60)

if __name__ == "__main__":
    run_phase_3()

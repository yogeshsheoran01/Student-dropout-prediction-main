import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import fetch_or_load_dataset
from src.preprocessing import preprocess_data
from src.model import train_logistic_regression, predict_student_risk

def run_phase_5():
    print("=" * 60)
    print("PHASE 5: LOGISTIC REGRESSION MODEL TRAINING")
    print("=" * 60)
    
    # 1. Load Dataset and Preprocess
    df = fetch_or_load_dataset(data_dir="data")
    prep_res = preprocess_data(df, target_col='Target', save_dir='models')
    
    X_train_scaled = prep_res['X_train_scaled']
    X_test_scaled = prep_res['X_test_scaled']
    y_train = prep_res['y_train']
    y_test = prep_res['y_test']
    feature_names = prep_res['feature_names']
    
    # 2. Train Logistic Regression Model
    model, coef_df = train_logistic_regression(
        X_train_scaled, y_train, feature_names, save_dir="models"
    )
    
    # 3. Model Predictions & Probabilities
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n[Training / Testing Split Details]:")
    print(f"  - Training Set: {len(y_train)} samples")
    print(f"  - Test Set:     {len(y_test)} samples")
    
    print("\n[Sample Predictions & Probabilities (First 10 Test Students)]:")
    sample_df = pd.DataFrame({
        'Student ID': range(1, 11),
        'True Class': y_test.values[:10],
        'Predicted Class': y_pred[:10],
        'Dropout Probability (%)': np.round(y_prob[:10] * 100, 2)
    })
    print(sample_df.to_string(index=False))
    
    print("\n[Top Risk Drivers (Positive Coefficients -> Increase Dropout Risk)]:")
    pos_drivers = coef_df[coef_df['Coefficient'] > 0].sort_values(by='Coefficient', ascending=False).head(5)
    for idx, row in pos_drivers.iterrows():
        print(f"  + {row['Feature']}: {row['Coefficient']:+.4f}")
        
    print("\n[Top Protective Factors (Negative Coefficients -> Decrease Dropout Risk)]:")
    neg_drivers = coef_df[coef_df['Coefficient'] < 0].sort_values(by='Coefficient', ascending=True).head(5)
    for idx, row in neg_drivers.iterrows():
        print(f"  - {row['Feature']}: {row['Coefficient']:+.4f}")
        
    # Visual Proof Generation for Phase 5
    os.makedirs("visuals", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Top 10 Coefficients Horizontal Bar Chart
    top_10 = coef_df.head(10).sort_values(by='Coefficient', ascending=True)
    colors = ['#d62728' if val > 0 else '#2ca02c' for val in top_10['Coefficient']]
    axes[0].barh(top_10['Feature'], top_10['Coefficient'], color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title('Top 10 Logistic Regression Coefficients\n(Red=Increases Risk, Green=Reduces Risk)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Model Coefficient (Log-Odds Impact)', fontsize=11)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    
    # 2. Prediction Probabilities Distribution (Test Set)
    axes[1].hist(y_prob[y_test == 0], bins=25, alpha=0.65, label='Retained Students (0)', color='#2ca02c', edgecolor='black')
    axes[1].hist(y_prob[y_test == 1], bins=25, alpha=0.65, label='Dropout Students (1)', color='#d62728', edgecolor='black')
    axes[1].set_title('Predicted Dropout Probabilities Distribution (Test Set)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted Dropout Probability', fontsize=11)
    axes[1].set_ylabel('Number of Students', fontsize=11)
    axes[1].axvline(0.30, color='orange', linestyle='--', linewidth=2, label='Moderate Risk Threshold (0.30)')
    axes[1].axvline(0.65, color='red', linestyle='--', linewidth=2, label='High Risk Threshold (0.65)')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("visuals/phase5_model_training.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[Screenshot Proof]: Saved Phase 5 training visual to 'visuals/phase5_model_training.png'")
    print("=" * 60)

if __name__ == "__main__":
    run_phase_5()

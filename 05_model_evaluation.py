import os
import joblib
import pandas as pd
import numpy as np
from src.dataset import fetch_or_load_dataset
from src.preprocessing import preprocess_data
from src.evaluation import evaluate_model

def run_phase_6():
    print("=" * 60)
    print("PHASE 6: MODEL EVALUATION & ERROR ANALYSIS")
    print("=" * 60)
    
    # Load dataset & preprocess
    df = fetch_or_load_dataset(data_dir="data")
    prep_res = preprocess_data(df, target_col='Target', save_dir='models')
    
    X_test_scaled = prep_res['X_test_scaled']
    y_test = prep_res['y_test']
    
    # Load saved Logistic Regression model
    model_path = os.path.join("models", "logistic_regression_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run 04_model_training.py first!")
        
    model = joblib.load(model_path)
    
    # Evaluate model
    metrics, report_text = evaluate_model(model, X_test_scaled, y_test, output_dir="visuals")
    
    print("\n[Summary Metrics Table]:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  - {k:<45}: {v:.4f} ({v*100:.2f}%)")
        else:
            print(f"  - {k:<45}: {v}")
            
    print("\n[Educational Early-Warning False Positive vs False Negative Analysis]:")
    fp = metrics['False Positives (Over-flagged as At-Risk)']
    fn = metrics['False Negatives (Missed Dropout Risk)']
    total_test = len(y_test)
    
    print(f"  1. False Positives ({fp} cases / {fp/total_test*100:.1f}%):")
    print("     - Meaning: Students predicted at-risk who actually complete their studies.")
    print("     - Impact: Advisor time allocated to safe students; minor resource overhead without harming student.")
    
    print(f"  2. False Negatives ({fn} cases / {fn/total_test*100:.1f}%):")
    print("     - Meaning: Students predicted safe who ultimately drop out unassisted.")
    print("     - Impact: High cost in educational early-warning systems; missed opportunity for critical early intervention.")
    print("     - Recommendation: Maintain a lower sensitivity threshold (e.g. 0.30) to catch potential false negatives early!")
    
    print("=" * 60)

if __name__ == "__main__":
    run_phase_6()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

def evaluate_model(model, X_test_scaled, y_test, output_dir="visuals"):
    """
    Evaluates Logistic Regression model performance with classification metrics,
    Confusion Matrix, ROC Curve, and False Positive/Negative analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Predictions and Probabilities
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'True Negatives (Retained correctly)': tn,
        'False Positives (Over-flagged as At-Risk)': fp,
        'False Negatives (Missed Dropout Risk)': fn,
        'True Positives (At-Risk Dropout correctly caught)': tp
    }
    
    # Plot Confusion Matrix & ROC Curve side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Confusion Matrix
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
        xticklabels=['Retained (0)', 'Dropout (1)'],
        yticklabels=['Retained (0)', 'Dropout (1)'],
        cbar=False, annot_kws={"size": 14, "weight": "bold"}
    )
    axes[0].set_title('Logistic Regression Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Logistic Regression (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    axes[1].set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    axes[1].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower right", fontsize=11)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    cm_roc_path = os.path.join(output_dir, "phase6_model_evaluation.png")
    plt.savefig(cm_roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print Classification Summary
    report_text = classification_report(y_test, y_pred, target_names=['Retained (0)', 'Dropout (1)'])
    print("=== MODEL EVALUATION REPORT ===")
    print(report_text)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Saved evaluation visualizations to {cm_roc_path}")
    
    return metrics, report_text

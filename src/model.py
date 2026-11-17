import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train_logistic_regression(X_train_scaled, y_train, feature_names, save_dir="models", C=1.0):
    """
    Trains a Logistic Regression model for student dropout risk prediction.
    Extracts feature coefficients to highlight top early-warning predictors.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize and fit Logistic Regression
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Save Model
    model_path = os.path.join(save_dir, "logistic_regression_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model successfully trained and saved to {model_path}")
    
    # Extract Feature Importance (Coefficients)
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    }).sort_values(by='Abs_Coefficient', ascending=False)
    
    return model, coef_df

def predict_student_risk(model, scaler, feature_names, input_dict):
    """
    Predicts dropout probability and risk category for a single student profile.
    
    Risk Tiers:
    - Low Risk: Probability < 0.30
    - Moderate Risk: 0.30 <= Probability <= 0.65
    - High Risk: Probability > 0.65
    """
    input_df = pd.DataFrame([input_dict])
    
    # Align features
    for f in feature_names:
        if f not in input_df.columns:
            input_df[f] = 0.0
            
    input_df = input_df[feature_names]
    
    # Scale input
    scaled_input = scaler.transform(input_df)
    
    # Probability prediction
    prob = model.predict_proba(scaled_input)[0][1]
    pred_class = int(model.predict(scaled_input)[0])
    
    if prob > 0.65:
        risk_level = "High Risk"
        risk_color = "red"
        recommendation = "Immediate Intervention Required: Schedule academic counseling, verify tuition support/financial aid, and enroll in structured peer tutoring."
    elif prob >= 0.30:
        risk_level = "Moderate Risk"
        risk_color = "orange"
        recommendation = "Monitor Progress: Flag for mid-term review, provide academic advising, and monitor class attendance."
    else:
        risk_level = "Low Risk"
        risk_color = "green"
        recommendation = "Standard Academic Track: Student demonstrates strong retention indicators. Continue standard periodic check-ins."
        
    return {
        'dropout_probability': float(prob),
        'predicted_class': pred_class,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'recommendation': recommendation
    }

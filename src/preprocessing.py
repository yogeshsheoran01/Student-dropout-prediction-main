import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(df, target_col='Target', save_dir='models'):
    """
    Cleans and preprocesses student dropout dataset for Logistic Regression.
    
    1. Target binarization: 'Dropout' -> 1 (At Risk), 'Graduate'/'Enrolled' -> 0 (Retained)
    2. Missing value handling (imputation if needed)
    3. Feature scaling with StandardScaler
    4. Splitting into Train and Test sets
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Copy dataframe
    data = df.copy()
    
    # Standardize column names (strip trailing whitespace/tabs if any)
    data.columns = [col.strip() for col in data.columns]
    target_col = target_col.strip()
    
    # Missing Value Analysis
    missing_summary = data.isnull().sum()
    total_missing = missing_summary.sum()
    print(f"Missing Value Analysis: Total missing values = {total_missing}")
    if total_missing > 0:
        data = data.fillna(data.median(numeric_only=True))
    
    # Target Encoding (Binary Classification)
    # Dropout = 1, Graduate/Enrolled = 0
    if target_col in data.columns:
        data['Binary_Target'] = data[target_col].apply(lambda x: 1 if str(x).strip() == 'Dropout' else 0)
        y = data['Binary_Target']
        X = data.drop(columns=[target_col, 'Binary_Target'])
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(data.columns)}")
    
    # Ensure all feature columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    feature_names = list(X.columns)
    
    # Train-Test Split (80% Train, 20% Test stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler and Feature Names
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(save_dir, "feature_names.pkl"))
    
    preprocessed_data = {
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'missing_summary': missing_summary
    }
    
    return preprocessed_data

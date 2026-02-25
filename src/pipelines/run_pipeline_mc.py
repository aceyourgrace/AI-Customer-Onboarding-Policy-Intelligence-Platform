# -------------------------------
# run_pipeline_mc.py
# Multi-class ConversionClass Pipeline (Logistic Regression)
# -------------------------------


# Quick Fix for Module Imports
# -------------------------------
import sys
import os

# Add the src folder to sys.path so imports work from pipelines/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------------
# 0. Imports
# -------------------------------
import pandas as pd

# Feature modules
from features.select_features import select_features
from features.scale_features import scale_numeric_features
from features.one_hot_encoding import one_hot_encode

# Train-test split (custom for multiclass)
from multiclass_modeling.train_test_split_mc import split_data_mc

# Model training
from multiclass_modeling.train_lr_model import train_lr_model, evaluate_lr_model

# -------------------------------
# 1. Load Dataset
# -------------------------------
file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"

# Dataset is tab-separated
df = pd.read_csv(file_path, sep='\t')
print(f"Dataset shape: {df.shape}")

# -------------------------------
# 2. Feature Selection
# -------------------------------
# Automatically select numeric and categorical features
# Set target_col='ConversionClass' for multiclass
X, y, numeric_features, categorical_features = select_features(df, target_col='ConversionClass')

print(f"Features selected: {len(X.columns)}")
print("Target distribution:")
print(y.value_counts(normalize=True))

# -------------------------------
# 3. One-Hot Encoding
# -------------------------------
X_encoded = one_hot_encode(X, categorical_features)
print(f"Shape after one-hot encoding: {X_encoded.shape}")

# -------------------------------
# 4. Train-Test Split
# -------------------------------
# Use custom split to allow stratification for class imbalance
X_train, X_test, y_train, y_test = split_data_mc(
    X_encoded,
    y,
    test_size=0.2,
    stratify=y  # preserves class distribution
)
print(f"Shape X_train: {X_train.shape}, X_test: {X_test.shape}")

# -------------------------------
# 5. Scaling Numeric Features
# -------------------------------
X_train, scaler = scale_numeric_features(X_train, numeric_features, fit=True)
X_test, _ = scale_numeric_features(X_test, numeric_features, scaler=scaler, fit=False)

# -------------------------------
# 6. Train Multi-Class Logistic Regression
# -------------------------------
model = train_lr_model(X_train, y_train)
print("Model training complete!")

# -------------------------------
# 7. Evaluate Model
# -------------------------------
accuracy, report = evaluate_lr_model(model, X_test, y_test)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
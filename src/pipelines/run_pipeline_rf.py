# -------------------------------
# run_pipeline_rf.py
# Multi-class ConversionClass Pipeline using Random Forest
# -------------------------------

# Quick Fix for Module Imports
# -------------------------------
import sys
import os

# Add the src folder to sys.path so imports work from pipelines/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Feature & model modules
from features.select_features import select_features
from features.one_hot_encoding import one_hot_encode
from features.scale_features import scale_numeric_features

# Train-test split
from multiclass_modeling.train_test_split_mc import split_data_mc

# Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# -------------------------------
# Step 0: File path to your raw CSV
# -------------------------------
import pandas as pd

file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv(file_path, sep='\t')
print(f"Dataset shape: {df.shape}")

# -------------------------------
# Step 2: Feature Selection
# -------------------------------
X, _, numeric_features, categorical_features = select_features(df)
y = df['ConversionClass']

print(f"Features selected: {len(X.columns)}")
print("Target distribution:")
print(y.value_counts(normalize=True))

# -------------------------------
# Step 3: One-Hot Encoding
# -------------------------------
X_encoded = one_hot_encode(X, categorical_features)
print(f"Shape after one-hot encoding: {X_encoded.shape}")

# -------------------------------
# Step 4: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = split_data_mc(
    X_encoded,
    y,
    test_size=0.2
)
print(f"Shape X_train: {X_train.shape}, X_test: {X_test.shape}")

# -------------------------------
# Step 5: Scaling numeric features
# -------------------------------
X_train, scaler = scale_numeric_features(
    X_train,
    numeric_features,
    fit=True
)

X_test, _ = scale_numeric_features(
    X_test,
    numeric_features,
    scaler=scaler,
    fit=False
)

# -------------------------------
# Step 6: Train Random Forest
# -------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight='balanced'  # helps with class imbalance
)
rf_model.fit(X_train, y_train)
print("Random Forest training complete!")

# -------------------------------
# Step 7: Evaluate Model
# -------------------------------
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(cm)
# -------------------------------
# run_hpt_grid_rf.py
# Random Forest with GridSearchCV for Multi-Class Leads
# Continuation from RandomizedSearchCV step
# -------------------------------

# Quick Fix for Module Imports
# -------------------------------
import sys
import os

# Add the src folder to sys.path so imports work from pipelines/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"
df = pd.read_csv(file_path, sep='\t')

# -------------------------------
# 2️⃣ Feature Selection & Target
# -------------------------------
from features.select_features import select_features
from features.one_hot_encoding import one_hot_encode
from features.scale_features import scale_numeric_features
from multiclass_modeling.train_test_split_mc import split_data_mc

X, _, numeric_features, categorical_features = select_features(df)
y = df['ConversionClass']

# -------------------------------
# 3️⃣ One-Hot Encode Categorical Features
# -------------------------------
X_encoded = one_hot_encode(X, categorical_features)

# -------------------------------
# 4️⃣ Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = split_data_mc(
    X_encoded, y, test_size=0.2
)

# -------------------------------
# 5️⃣ Scaling Numeric Features
# -------------------------------
X_train, scaler = scale_numeric_features(X_train, numeric_features, fit=True)
X_test, _ = scale_numeric_features(X_test, numeric_features, scaler=scaler, fit=False)

print(f"Shape X_train: {X_train.shape}, X_test: {X_test.shape}")

# -------------------------------
# 6️⃣ GridSearchCV for Random Forest
# -------------------------------
# Start with the best params from RandomizedSearchCV as base
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Fine-tune around the previous best hyperparameters
param_grid = {
    'n_estimators': [350, 400, 450],   # try a few values around the previous best (400)
    'max_depth': [None, 15, 20],       # allow deeper trees
    'min_samples_leaf': [1, 2, 3],     # fine-tune leaf size
    'max_features': ['sqrt', 'log2']   # check feature selection per split
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,            # 3-fold cross-validation
    verbose=2,
    n_jobs=-1
)

print("Starting GridSearchCV for Random Forest...")
grid_search.fit(X_train, y_train)
print("GridSearchCV complete!")

# -------------------------------
# 7️⃣ Evaluate Best Model
# -------------------------------
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nBest Hyperparameters from GridSearchCV:", grid_search.best_params_)
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)
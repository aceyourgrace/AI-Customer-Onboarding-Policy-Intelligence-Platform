# -------------------------------
# run_hpt_grid_xgb.py
# Hyperparameter tuning for XGBoost (multiclass)
# -------------------------------

# Quick Fix for Module Imports
# -------------------------------
import sys
import os

# Add the src folder to sys.path so imports work from pipelines/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from features.select_features import select_features
from features.one_hot_encoding import one_hot_encode
from features.scale_features import scale_numeric_features
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"
df = pd.read_csv(file_path, sep='\t')

# -------------------------------
# Step 2: Feature Selection
# -------------------------------
X, _, numeric_features, categorical_features = select_features(df)
y = df['ConversionClass']

# -------------------------------
# Step 3: Encode categorical features
# -------------------------------
X_encoded = one_hot_encode(X, categorical_features)

# -------------------------------
# Step 4: Label encode target (required for XGBoost)
# -------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # High=0, Low=1, Medium=2 (or similar)
# Store mapping for later reference:
print("Target mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# -------------------------------
# Step 5: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# -------------------------------
# Step 6: Scale numeric features
# -------------------------------
X_train, scaler = scale_numeric_features(X_train, numeric_features, fit=True)
X_test, _ = scale_numeric_features(X_test, numeric_features, scaler=scaler, fit=False)

print(f"Shape X_train: {X_train.shape}, X_test: {X_test.shape}")

# -------------------------------
# Step 7: GridSearchCV for XGBoost
# -------------------------------
xgb = XGBClassifier(
    objective='multi:softmax',  # multiclass
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# Grid of hyperparameters to try
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 5, 7, None],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

# -------------------------------
# Step 8: Fit GridSearch
# -------------------------------
grid_search.fit(X_train, y_train)
print("GridSearchCV complete!\n")
print("Best Hyperparameters from GridSearchCV:", grid_search.best_params_)

# -------------------------------
# Step 9: Evaluate Best Model
# -------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(cm)

import joblib
import os

# Create models folder if it doesn't exist
models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
os.makedirs(models_dir, exist_ok=True)

# Save the trained model
joblib.dump(best_model, os.path.join(models_dir, 'xgb_best_model.pkl'))

# Save the scaler
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

# Save the label encoder
joblib.dump(le, os.path.join(models_dir, 'label_encoder.pkl'))

# Optionally, save the list of categorical feature columns after one-hot encoding
encoded_columns = X_encoded.columns.tolist()
joblib.dump(encoded_columns, os.path.join(models_dir, 'encoded_columns.pkl'))

print("All models and transformers saved to models/ folder!")
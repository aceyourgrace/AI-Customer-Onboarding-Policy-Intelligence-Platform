# -------------------------------
# predict_new_leads_xg.py
# Predict new leads using saved XGBoost model
# -------------------------------

import sys
import os

# -------------------------------
# Fix imports: Add src/ to Python path
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # src/

# Now import modules
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from features.select_features import select_features
from features.one_hot_encoding import one_hot_encode
from features.scale_features import scale_numeric_features

# -------------------------------
# Step 1: Set file paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to new leads CSV (input)
new_data_path = os.path.join(BASE_DIR, '..', '..', 'data', 'raw', 'new_leads.csv')

# Path to saved XGBoost model
model_path = os.path.join(BASE_DIR, '..', '..', 'models', 'xgb_best_model.pkl')

# Path to saved scaler
scaler_path = os.path.join(BASE_DIR, '..', '..', 'models', 'scaler.pkl')

# Path to save predictions
output_path = os.path.join(BASE_DIR, '..', '..', 'data', 'processed', 'new_leads_predictions_xg.csv')

# -------------------------------
# Step 2: Load new leads dataset
# -------------------------------
df_new = pd.read_csv(new_data_path)
print(f"Loaded new leads dataset with shape: {df_new.shape}")

# -------------------------------
# Step 3: Load saved model and scaler
# -------------------------------
xgb_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("Model and scaler loaded successfully!")

# -------------------------------
# Step 4: Feature selection & encoding
# -------------------------------
# Use same select_features logic as training
X_new, _, numeric_features, categorical_features = select_features(df_new, is_training=False)

# One-hot encode categorical features
X_new_encoded = one_hot_encode(X_new, categorical_features)

# Scale numeric features
X_new_encoded, _ = scale_numeric_features(X_new_encoded, numeric_features, scaler=scaler, fit=False)

# -------------------------------
# Step 5: Predict with XGBoost
# -------------------------------
y_pred_encoded = xgb_model.predict(X_new_encoded)

# Load the LabelEncoder mapping from training
# For simplicity, we'll manually define it same as training:
target_mapping = {'High': 0, 'Low': 1, 'Medium': 2}
inv_target_mapping = {v: k for k, v in target_mapping.items()}

# Convert numeric prediction back to original labels
y_pred = [inv_target_mapping[i] for i in y_pred_encoded]

# -------------------------------
# Step 6: Save predictions
# -------------------------------
df_new['Predicted_ConversionClass'] = y_pred
df_new.to_csv(output_path, index=False)

print(f"Predictions saved to: {output_path}")
print(df_new.head())
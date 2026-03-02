# -------------------------------
# predict_cltv_new_leads.py
# Predict CLTV for new leads using trained RandomForest model
# -------------------------------

import sys
import os
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Step 0: Fix imports
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -------------------------------
# Step 1: Paths
# -------------------------------
new_leads_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'new_leads.csv')
)

cltv_model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'cltv_RandomForest.pkl')
)

conv_class_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'new_leads_predictions_xg.csv')
)

output_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'new_leads_with_cltv.csv')
)

# -------------------------------
# Step 2: Load new leads
# -------------------------------
df_new = pd.read_csv(new_leads_path)
print(f"Loaded new leads dataset with shape: {df_new.shape}")

# -------------------------------
# Step 3: Load CLTV model
# -------------------------------
cltv_model = joblib.load(cltv_model_path)
print("CLTV RandomForest model loaded successfully!")

# -------------------------------
# Step 4: Predict CLTV
# -------------------------------
# Pass raw df; pipeline handles preprocessing
df_new['Predicted_CLTV'] = cltv_model.predict(df_new)
print("CLTV prediction complete!")

# -------------------------------
# Step 5: Merge conversion class safely
# -------------------------------
if os.path.exists(conv_class_path):
    df_conv = pd.read_csv(conv_class_path)
    if 'ConversionClass' in df_conv.columns:
        df_new['ConversionClass'] = df_conv['ConversionClass']
        print("ConversionClass merged successfully!")

# -------------------------------
# Step 6: Normalize CLTV Score
# -------------------------------
min_val = df_new['Predicted_CLTV'].min()
max_val = df_new['Predicted_CLTV'].max()

df_new['CLTV_Score_Normalized'] = (
    (df_new['Predicted_CLTV'] - min_val) / (max_val - min_val)
)

# -------------------------------
# Step 7: Multi-Level Priority Segmentation
# -------------------------------
priority_bins = [0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0]
priority_labels = [
    "Very Low Priority",
    "Low Priority",
    "Lower-Medium Priority",
    "Medium Priority",
    "High Priority",
    "Very High Priority"
]

df_new['Lead_Priority'] = pd.cut(
    df_new['CLTV_Score_Normalized'],
    bins=priority_bins,
    labels=priority_labels,
    include_lowest=True
)

print("\nPriority Distribution:")
print(df_new['Lead_Priority'].value_counts())

# -------------------------------
# Step 8: Keep all useful columns for AI agent
# -------------------------------
useful_columns = [
    'LeadID',
    'Age',
    'Gender',
    'Province',
    'EmploymentStatus',
    'MaritalStatus',
    'InitialProductInterest',
    'LeadSource',
    'FirstContactChannel',
    'ConversionClass',
    'Predicted_CLTV',
    'CLTV_Score_Normalized',
    'Lead_Priority'
]

df_final = df_new.copy()
missing_cols = [col for col in useful_columns if col not in df_final.columns]
if missing_cols:
    print("\nWarning: Missing columns:", missing_cols)

df_final = df_final[[col for col in useful_columns if col in df_final.columns]]

# -------------------------------
# Step 9: Save enriched dataset
# -------------------------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_final.to_csv(output_path, index=False)

print(f"\nSaved FULL enriched dataset to {output_path}")
print(f"Final dataset shape: {df_final.shape}")
print("\nColumns:")
print(df_final.columns.tolist())
print("\nSample preview:")
print(df_final.head())
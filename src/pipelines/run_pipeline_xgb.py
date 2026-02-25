# -------------------------------
# run_pipeline_xgb.py
# Multi-class ConversionClass Pipeline using XGBoost
# -------------------------------

# Quick Fix for Module Imports
# -------------------------------
import sys
import os

# Add the src folder to sys.path so imports work from pipelines/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -------------------------------
# Imports
# -------------------------------
import pandas as pd
from features.select_features import select_features
from features.one_hot_encoding import one_hot_encode
from features.scale_features import scale_numeric_features
from multiclass_modeling.train_test_split_mc import split_data_mc
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# File path to your raw CSV
# -------------------------------
file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv(file_path, sep='\t')  # tab-separated

print(f"Dataset shape: {df.shape}")

# -------------------------------
# Step 2: Feature Selection
# -------------------------------
X, y, numeric_features, categorical_features = select_features(df)

# Overwrite target to ConversionClass
y = df['ConversionClass']
print(f"Features selected: {len(X.columns)}")
print(f"Target distribution:\n{y.value_counts(normalize=True)}")

# -------------------------------
# Step 3: One-Hot Encoding categorical features
# -------------------------------
X_encoded = one_hot_encode(X, categorical_features)
print(f"Shape after one-hot encoding: {X_encoded.shape}")

# -------------------------------
# Step 4: Encode target for XGBoost
# -------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Converts ['High','Medium','Low'] -> [0,1,2]
print("Target classes after encoding:", le.classes_)

# -------------------------------
# Step 5: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = split_data_mc(
    X_encoded, y_encoded, test_size=0.2
)
print(f"Shape X_train: {X_train.shape}, X_test: {X_test.shape}")

# -------------------------------
# Step 6: Scaling numeric features
# -------------------------------
X_train, scaler = scale_numeric_features(X_train, numeric_features, fit=True)
X_test, _ = scale_numeric_features(X_test, numeric_features, scaler=scaler, fit=False)

# -------------------------------
# Step 7: Train XGBoost Classifier
# -------------------------------
xgb_model = XGBClassifier(
    objective='multi:softmax',       # multiclass classification
    num_class=len(le.classes_),      # number of classes
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)
print("XGBoost training complete!")

# -------------------------------
# Step 8: Evaluate Model
# -------------------------------
y_pred = xgb_model.predict(X_test)

# Map predictions back to original labels for readability
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

accuracy = accuracy_score(y_test_labels, y_pred_labels)
report = classification_report(y_test_labels, y_pred_labels)
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)
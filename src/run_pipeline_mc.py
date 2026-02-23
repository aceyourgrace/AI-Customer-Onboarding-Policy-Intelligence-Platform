

# -------------------------------
# run_pipeline_mc.py
# Multi-class ConversionClass Pipeline
# -------------------------------

# Feature & model modules
from features.select_features import select_features
from features.one_hot_encoding import one_hot_encode
from features.scale_features import scale_numeric_features

# Train-test split and evaluation
from binary_modeling.train_test_split import split_data  # reuse existing function
from sklearn.metrics import classification_report, confusion_matrix


# File path to your raw CSV
file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"
# -------------------------------
# Step 1: Load Dataset
# -------------------------------
import pandas as pd

df = pd.read_csv(file_path, sep='\t')  # remember your dataset is tab-separated

# -------------------------------
# Step 2: Feature Selection
# -------------------------------
# Reuse your select_features function
X, _, numeric_features, categorical_features = select_features(df)

# Overwrite target for multi-class
y = df['ConversionClass']

# print(f"Dataset shape: {df.shape}")
# print(f"Features selected: {len(X.columns)}")
# print(f"Target distribution:\n{y.value_counts(normalize=True)}")

# -------------------------------
# Step 3: One-Hot Encode categorical features
# -------------------------------
X_encoded = one_hot_encode(X, categorical_features)

# print(f"Shape after one-hot encoding: {X_encoded.shape}")

from multiclass_modeling.train_test_split_mc import split_data_mc

# -------------------------------
# Step 4: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = split_data_mc(
    X_encoded,
    y,
    test_size=0.2
)

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

print(f"Shape X_train: {X_train.shape}, X_test: {X_test.shape}")

# -------------------------------
# Step 6: Train Multi-Class Logistic Regression
# -------------------------------
from multiclass_modeling.train_lr_model import train_lr_model

model = train_lr_model(X_train, y_train)
print("Model training complete!")

from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# Step 7: Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
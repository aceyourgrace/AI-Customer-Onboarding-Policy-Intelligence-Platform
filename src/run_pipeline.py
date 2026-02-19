

from load_data import load_dataset
from features.select_features import select_features
from features.one_hot_encoding import one_hot_encode
from features.scale_features import scale_numeric_features
from modeling.train_test_split import split_data
from modeling.train_model import train_model, evaluate_model

file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"

# Load dataset
df = load_dataset(file_path)

# Select features
X, y, numeric_features, categorical_features = select_features(df)

# One-hot encode categorical features
X_encoded = one_hot_encode(X, categorical_features)
# print("Final Shape:", X_encoded.shape)

# Scale numeric features
X_scaled = scale_numeric_features(X_encoded, numeric_features)

# print("Pipeline complete. Final shape:", X_scaled.shape)

# Split into train and test sets
X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=0.2)

# print("Shapes after train-test split:")
# print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
# print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# Train model
model = train_model(X_train, y_train)

# Evaluate model
accuracy, report = evaluate_model(model, X_test, y_test)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)


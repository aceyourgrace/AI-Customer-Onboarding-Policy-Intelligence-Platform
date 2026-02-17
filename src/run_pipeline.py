




from load_data import load_dataset
from features.select_features import select_features
from features.one_hot_encoding import one_hot_encode
from features.scale_features import scale_numeric_features

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

print("Pipeline complete. Final shape:", X_scaled.shape)





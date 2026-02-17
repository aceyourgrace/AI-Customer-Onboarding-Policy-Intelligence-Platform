
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_numeric_features(df, numeric_features):
    """
    Scale numeric features using StandardScaler.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing features
        numeric_features (list): List of numeric columns to scale
    
    Returns:
        pd.DataFrame: DataFrame with scaled numeric columns (categorical remain unchanged)
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])
    
    return df_scaled

# if __name__ == "__main__":
#     from load_data import load_dataset
#     from features.select_features import select_features
#     from features.one_hot_encoding import one_hot_encode

#     # Load dataset
#     file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"
#     df = load_dataset(file_path)

#     # Select features
#     X, y, numeric_features, categorical_features = select_features(df)

#     # Encode categorical features
#     X_encoded = one_hot_encode(X, categorical_features)

#     # Scale numeric features
#     X_scaled = scale_numeric_features(X_encoded, numeric_features)

#     print("Final shape after scaling:", X_scaled.shape)
#     print("Preview of scaled data:\n", X_scaled.head())


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(df, categorical_features):
    """
    One-hot encode the given categorical columns.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing features
        categorical_features (list): List of categorical column names to encode
    
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns
    """
    encoder = OneHotEncoder(sparse_output=False, drop='first') # drop='first' avoids redundant columns
    encoded_array = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_features))
    
    # Keep numeric columns unchanged
    numeric_cols = [col for col in df.columns if col not in categorical_features]
    final_df = pd.concat([df[numeric_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    
    return final_df


if __name__ == "__main__":
    # Example usage
    from features.select_features import select_features
    from load_data import load_dataset

    file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"
    df = load_dataset(file_path)

    X, y, numeric_features, categorical_features = select_features(df)
    X_encoded = one_hot_encode(X, categorical_features)
    
    print("Shape after one-hot encoding:", X_encoded.shape)
    print("Columns after encoding:", X_encoded.columns)

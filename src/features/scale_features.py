
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_numeric_features(df, numeric_features, scaler=None, fit=False):
    """
    Scale numeric features using StandardScaler.

    Parameters:
        df (pd.DataFrame): DataFrame containing features
        numeric_features (list): List of numeric columns to scale
        scaler (StandardScaler): Existing scaler (optional)
        fit (bool): Whether to fit the scaler

    Returns:
        df_scaled (pd.DataFrame)
        scaler (StandardScaler)
    """

    df_scaled = df.copy()

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])
    else:
        df_scaled[numeric_features] = scaler.transform(df_scaled[numeric_features])

    return df_scaled, scaler
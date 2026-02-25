import pandas as pd
from sklearn.preprocessing import StandardScaler

def select_features(df, target_col='ConvertedFlag'):
    """
    Selects numeric and categorical features and separates the target.

    Parameters:
        df (pd.DataFrame): DataFrame containing raw data
        target_col (str): Column name of the target variable

    Returns:
        X (pd.DataFrame): Feature DataFrame
        y (pd.Series): Target column
        numeric_features (list): List of numeric features
        categorical_features (list): List of categorical features
    """
    # -------------------------
    # Define numeric & categorical features
    # -------------------------
    numeric_features = [
        'Age', 'Income', 'WebsiteVisits_PreConversion', 'TimeOnWebsite_Minutes',
        'EmailOpenedCount', 'DaysSinceInquiry', 'CallCenterInquiries', 'BranchVisits',
        'CLTV', 'EngagementScore', 'AgeScore', 'LeadPriorityScore', 'FirstTransactionAmount',
        'MonthlyRevenue', 'TenureMonths'
    ]

    categorical_features = [
        'Gender', 'EmploymentStatus', 'MaritalStatus', 'Province',
        'InitialProductInterest', 'FirstContactChannel', 'LeadSource',
        'HighValueLead', 'ReferrerCustomer', 'HasExistingProducts'
    ]

    # -------------------------
    # Separate target
    # -------------------------
    y = df[target_col]
    
    # Features (X)
    X = df[numeric_features + categorical_features].copy()
    
    return X, y, numeric_features, categorical_features


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
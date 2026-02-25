import pandas as pd


def select_features(df, target_col='ConvertedFlag', is_training=True):
    """
    Select features and (optionally) target variable.

    Parameters:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        is_training (bool): 
            True  -> expect target column (training mode)
            False -> do NOT expect target column (prediction mode)

    Returns:
        X (pd.DataFrame): feature dataframe
        y (pd.Series or None): target variable (None if prediction mode)
        numeric_features (list)
        categorical_features (list)
    """

    # -----------------------------
    # Feature Lists
    # -----------------------------
    numeric_features = [
        'Age', 'Income', 'WebsiteVisits_PreConversion',
        'TimeOnWebsite_Minutes', 'EmailOpenedCount',
        'DaysSinceInquiry', 'CallCenterInquiries',
        'BranchVisits', 'CLTV', 'EngagementScore',
        'AgeScore', 'LeadPriorityScore'
    ]

    categorical_features = [
        'Gender', 'EmploymentStatus', 'MaritalStatus',
        'Province', 'InitialProductInterest',
        'FirstContactChannel', 'LeadSource'
    ]

    # -----------------------------
    # Validate Columns
    # -----------------------------
    required_features = numeric_features + categorical_features

    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    # -----------------------------
    # Select Features
    # -----------------------------
    X = df[required_features].copy()

    # -----------------------------
    # Handle Target (Training vs Prediction)
    # -----------------------------
    if is_training:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")

        y = df[target_col]
    else:
        y = None

    return X, y, numeric_features, categorical_features
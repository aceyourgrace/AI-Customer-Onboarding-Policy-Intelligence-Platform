
#Step 6.5:

#6.5 - 1:

# First, lets load the data again here cause this is a new python file

# import pandas as pd

# # Load the dataset
# file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"
# df = pd.read_csv(file_path, sep='\t')

# # Quick check
# print(df.head())
# print("\nDataset shape:", df.shape)


# # Lets select X and Y now
# y = df['ConvertedFlag']

# # Numeric features
# numeric_features = ['Age', 'Income', 'WebsiteVisits_PreConversion', 'TimeOnWebsite_Minutes',
#                     'EmailOpenedCount', 'DaysSinceInquiry', 'CallCenterInquiries', 'BranchVisits',
#                     'CLTV', 'EngagementScore', 'AgeScore', 'LeadPriorityScore']

# # Categorical features
# categorical_features = ['Gender', 'EmploymentStatus', 'MaritalStatus', 'Province', 
#                         'InitialProductInterest', 'FirstContactChannel', 'LeadSource']

# # Combine into a feature dataframe
# X = df[numeric_features + categorical_features]

# print("Feature dataframe shape:", X.shape)
# print("Target variable shape:", y.shape)

# #6.5 - 2:

# # One-hot encode categorical features
# X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# print("Encoded feature dataframe shape:", X_encoded.shape)


# src/prep_features.py

import pandas as pd

def select_features(df):
    """
    Select numeric and categorical features from the dataframe.
    Returns X (features) and y (target).
    """
    # Target
    y = df['ConvertedFlag']

    # Numeric features
    numeric_features = [
        'Age', 'Income', 'WebsiteVisits_PreConversion', 'TimeOnWebsite_Minutes',
        'EmailOpenedCount', 'DaysSinceInquiry', 'CallCenterInquiries', 'BranchVisits',
        'CLTV', 'EngagementScore', 'AgeScore', 'LeadPriorityScore'
    ]

    # Categorical features
    categorical_features = [
        'Gender', 'EmploymentStatus', 'MaritalStatus', 'Province', 
        'InitialProductInterest', 'FirstContactChannel', 'LeadSource'
    ]

    # Combine numeric + categorical features
    X = df[numeric_features + categorical_features]

    return X, y, numeric_features, categorical_features


# def quick_feature_check(X, y):
#     """
#     Print info about features and target variable.
#     """
#     print("Feature dataframe shape:", X.shape)
#     print("Target variable shape:", y.shape)
#     print("\nFeature columns:\n", X.columns)
#     print("\nTarget name:", y.name)


# if __name__ == "__main__":
#     # Example usage if running this file directly
#     from load_data import load_dataset  # reuse our previous function

#     file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"
#     df = load_dataset(file_path)

#     # Prepare features
#     X, y, numeric_features, categorical_features = select_features(df)

#     # Quick check
#     quick_feature_check(X, y)

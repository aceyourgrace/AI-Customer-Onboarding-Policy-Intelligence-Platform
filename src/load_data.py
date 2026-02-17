
import pandas as pd

#Step 6.4:

# # Load the dataset with tab separator
# file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"
# df = pd.read_csv(file_path, sep='\t')  # <--- notice sep='\t'

# # Quick check
# print(df.head())
# print("\nDataset shape:", df.shape)

# # Check column names
# print("Columns in dataset:")
# print(df.columns)

# # Check data types
# print("\nData types:")
# print(df.dtypes)

# # Count missing values
# print("\nMissing values per column:")
# print(df.isnull().sum())

# #Summary Stats
# print("\nSummary statistics:")
# print(df[['Age', 'Income', 'MonthlyRevenue', 'CLTV', 'LeadPriorityScore']].describe())


# src/load_data.py

import pandas as pd

def load_dataset(file_path):
    """
    Load the CSV dataset into a pandas DataFrame.
    Handles tab-separated values.
    """
    df = pd.read_csv(file_path, sep="\t")  # important: \t for tab delimiter
    return df

def quick_check(df):
    """
    Print basic info about the dataset: head, shape, columns, dtypes, missing values, stats.
    """
    print("First 5 rows:\n", df.head(), "\n")
    print("Dataset shape:", df.shape, "\n")
    print("Columns in dataset:\n", df.columns, "\n")
    print("Data types:\n", df.dtypes, "\n")
    print("Missing values per column:\n", df.isnull().sum(), "\n")
    print("Summary statistics:\n", df.describe(), "\n")

if __name__ == "__main__":
    # Path to your dataset
    file_path = r"D:\ACEYOURGRACE\DATASCIENCE\DataScienceProjects\AI Customer Onboarding & Policy Intelligence Platform\bank-lead-intelligence\data\raw\bank_leads_v4.csv"
    
    # Load dataset
    df = load_dataset(file_path)
    
    # Quick check
    quick_check(df)



from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets.
    
    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        test_size (float): Proportion of dataset for testing
        random_state (int): Seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

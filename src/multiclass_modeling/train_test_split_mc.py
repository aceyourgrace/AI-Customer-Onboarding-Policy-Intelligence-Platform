
from sklearn.model_selection import train_test_split

def split_data_mc(X, y, test_size=0.2, random_state=42):
    """
    Splits dataset into train/test sets for multi-class tasks.
    Uses stratified split to preserve class proportions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # preserves High/Medium/Low proportions
    )
    return X_train, X_test, y_train, y_test
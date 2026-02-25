"""
train_rf_model.py

This module trains a Random Forest classifier
for the Multi-Class Lead Potential Model.

Classes:
    - High
    - Medium
    - Low

Author: Bikesh Chipalu
Project: Bank Lead Intelligence System
"""

from sklearn.ensemble import RandomForestClassifier


def train_rf_model(X_train, y_train):
    """
    Trains a Random Forest model on the training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training
    y_train : pd.Series
        Target labels (High / Medium / Low)

    Returns
    -------
    model : RandomForestClassifier
        Trained Random Forest model
    """

    # Initialize Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=200,          # Number of decision trees in the forest
        max_depth=None,            # Allow trees to grow fully (no depth restriction)
        min_samples_split=2,       # Minimum samples required to split a node
        min_samples_leaf=1,        # Minimum samples required at a leaf node
        random_state=42,           # Ensures reproducibility
        class_weight="balanced"    # Adjusts weights to handle class imbalance
    )

    # Train the model
    model.fit(X_train, y_train)

    return model
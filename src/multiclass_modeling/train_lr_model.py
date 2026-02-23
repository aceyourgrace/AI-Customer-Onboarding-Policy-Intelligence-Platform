from sklearn.linear_model import LogisticRegression

def train_lr_model(X_train, y_train):
    """
    Train a multi-class Logistic Regression model.
    """
    model = LogisticRegression(
        multi_class='multinomial',  # allows 3 classes
        solver='lbfgs',             # recommended for multinomial
        max_iter=500,
        class_weight='balanced',    # handles imbalanced classes
        random_state=42
    )

    model.fit(X_train, y_train)
    return model
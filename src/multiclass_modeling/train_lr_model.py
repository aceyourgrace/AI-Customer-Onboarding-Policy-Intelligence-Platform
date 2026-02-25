from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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


def evaluate_lr_model(model, X_test, y_test):
    """
    Evaluate the model and return accuracy & classification report
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report
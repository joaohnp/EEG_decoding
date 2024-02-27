"""Module to apply a simple XGBoost classifier to distinguish ERPs per condition

    Returns
    -------
    accuracy
        returns the model effectivness in distinguishing the condition
    """
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


def XGB_classification(X, y):
    """Applies XGBoost to the fed in data

Parameters
----------
X_train : np array
    ERPs from both conditions
y_train : binary
    binary discrimination per condition
X_test : np array
    ERPs to be tested
y_test : binary
    binary discrimination to be tested

Returns
-------
scores
    accuracy of the model
    """
# Initialize and train XGBoost
    xgb_clf = XGBClassifier()

    # Perform cross-validation and compute accuracy
    scores = cross_val_score(xgb_clf, X, y, cv=5, scoring='accuracy')

    # Calculate the mean and standard deviation of the cross-validation scores
    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    print(f'XGBoost Cross-Validation Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}')
    
    return scores
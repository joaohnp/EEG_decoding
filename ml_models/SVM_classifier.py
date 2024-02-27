"""Module to apply a simple SVM to classifiy ERPs per condition

    Returns
    -------
    accuracy
        returns the model effectivness in distinguishing the condition
    """
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC  # Support Vector Classifier


def SVM_classification(X_train, y_train, X_test, y_test):
    """Applies SVM to the fed in data

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
    # Initialize the SVM classifier
    # Kernel can be 'linear', 'poly', 'rbf', 'sigmoid', or a custom kernel function
    svm_clf = SVC(kernel='rbf')

    # Train the classifier
    svm_clf.fit(X_train, y_train)

    # Compute accuracy using cross-validation
    scores = cross_val_score(svm_clf, X_test, y_test, cv=5)
    print(f'SVM Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}')

    return scores

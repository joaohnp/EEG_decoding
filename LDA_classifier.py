"""Module to apply a simple RNN to classifiy ERPs per condition

    Returns
    -------
    accuracy
        returns the model effectivness in distinguishing the condition
    """
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score


def LDA_classification(X_train, y_train, X_test, y_test):
    """Applies LDA to the fed in data

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
    accuracy
        accuracy of the model
    """
    clf = LDA()
    clf.fit(X_train.reshape(X_train.shape[0], -1), y_train)  # Reshape for sklearn

    # Evaluate the classifier
    scores = cross_val_score(clf, X_test.reshape(X_test.shape[0], -1), y_test, cv=5)
    print(f'Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}')

    return scores
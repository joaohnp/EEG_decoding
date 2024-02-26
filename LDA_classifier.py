import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score


def LDA_classification(X_train, y_train, X_test, y_test):
    clf = LDA()
    clf.fit(X_train.reshape(X_train.shape[0], -1), y_train)  # Reshape for sklearn

    # Evaluate the classifier
    scores = cross_val_score(clf, X_test.reshape(X_test.shape[0], -1), y_test, cv=5)
    print(f'Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}')

    return scores
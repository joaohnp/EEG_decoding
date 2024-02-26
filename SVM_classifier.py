import numpy as np
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.model_selection import cross_val_score

def SVM_classification(X_train, y_train, X_test, y_test):
    # Initialize the SVM classifier
    # Kernel can be 'linear', 'poly', 'rbf', 'sigmoid', or a custom kernel function
    svm_clf = SVC(kernel='linear')

    # Train the classifier
    svm_clf.fit(X_train, y_train)

    # Compute accuracy using cross-validation
    scores = cross_val_score(svm_clf, X_test, y_test, cv=5)
    print(f'SVM Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}')

    return scores


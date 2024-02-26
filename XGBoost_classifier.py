import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


def XGB_classification(X, y):
# Initialize and train XGBoost
    xgb_clf = XGBClassifier()

    # Perform cross-validation and compute accuracy
    scores = cross_val_score(xgb_clf, X, y, cv=5, scoring='accuracy')

    # Calculate the mean and standard deviation of the cross-validation scores
    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    print(f'XGBoost Cross-Validation Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}')
    
    return scores
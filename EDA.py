#%%
import os
import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne import combine_evoked
from mne.datasets.limo import load_data
from mne.stats import linear_regression
from mne.viz import plot_compare_evokeds, plot_events
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Support Vector Classifier
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.layers.experimental.preprocessing import Normalization

# Build CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from xgboost import XGBClassifier

from LDA_classifier import LDA_classification
from RNN_classifier import RNN_classification
from SVM_classifier import SVM_classification
from XGBoost_classifier import XGB_classification

#%% Only one subject in this dataset
subj = 1
limo_epochs = load_data(subject=subj)
#   %% Plotting topoplot + ERPs, to better grasp the data. 
ts_args = dict(xlim=(-0.25, 0.5))
times_ofinterest = [0.09, 0.15, 0.23]

limo_epochs["Face/A"].average().plot_joint(
    times=times_ofinterest, title="Face A", ts_args=ts_args
)
limo_epochs["Face/B"].average().plot_joint(
    times=times_ofinterest, title="Face B", ts_args=ts_args
)
#   %% Grabbing the 'good' channels.
all_ch_names = limo_epochs.info['ch_names']
bad_ch_names = limo_epochs.info['bads']
ch_names = [ch for ch in all_ch_names if ch not in bad_ch_names]
#   %% Separating into two different epochs object.
epochs_a = limo_epochs["Face/A"].crop(tmin=0, tmax=0.5)
epochs_b = limo_epochs["Face/B"].crop(tmin=0, tmax=0.5)
# In this dataset, faces shown were more or less blurred. To the end of making
# a classifier to distinguish which face was shown, let's focus on highly
# coheren ones (less blurred)

coherence_values = limo_epochs.metadata["phase-coherence"]
levels = sorted(coherence_values.unique())
labels = [f"{i:.2f}" for i in np.arange(0.0, 0.90, 0.05)]

labels_levels_dict = {label: level for label, level in zip(labels, levels)}

epochs_a_filtered = []
epochs_b_filtered = []
COHERENCE_START = 0.50  # defining the interval of coherence values to grab
COHERENCE_END = 0.85

level_range_start = labels_levels_dict[f"{COHERENCE_START:.2f}"]
level_range_end = labels_levels_dict[f"{COHERENCE_END:.2f}"]

epochs_a_filtered = epochs_a[epochs_a.metadata['phase-coherence'].
                             between(level_range_start, level_range_end)]
epochs_b_filtered = epochs_b[epochs_b.metadata['phase-coherence'].
                             between(level_range_start, level_range_end)]

# now we have cropped the highly coherent epochs into two conditions. trying 
# classifiers now:
#   %% Linear discriminant anaysis (LDA)
ch_scores = {}
for channel in ch_names:
    X = np.concatenate([epochs_a_filtered.get_data(picks=channel),
                        epochs_b_filtered.get_data(picks=channel)])  # Feature matrix
    y = np.concatenate([np.zeros(len(epochs_a_filtered)),
                        np.ones(len(epochs_b_filtered))])  # Labels

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
    if sum(sum(sum(X_train))) == 0:
        scores = 0
        ch_scores[channel] = scores
    else:
        scores = LDA_classification(X_train, y_train, X_test, y_test)
        ch_scores[channel] = scores

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "scores", "LDA.pickle")
with open(filepath, 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(ch_scores, file)
#   %% Suppor Vector Machine (SVM)
ch_scores = {}
for channel in ch_names:
    # Concatenate data from two conditions and extract features
    X = np.concatenate([epochs_a_filtered.get_data(picks=channel),
                        epochs_b_filtered.get_data(picks=channel)])

    y = np.concatenate([np.zeros(len(epochs_a_filtered)),
                        np.ones(len(epochs_b_filtered))])
    X = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)

    if sum(sum(X_train)) == 0:
        scores = 0
        ch_scores[channel] = scores
    else:
        scores = SVM_classification(X_train, y_train, X_test, y_test)
        ch_scores[channel] = scores

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "scores", "SVM.pickle")
with open(filepath, 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(ch_scores, file)

#   %% XGVBoost classifier
ch_scores = {}
for channel in ch_names:
    # Concatenate data from two conditions and extract features
    X = np.concatenate([epochs_a_filtered.get_data(picks=channel), 
                        epochs_b_filtered.get_data(picks=channel)])
    y = np.concatenate([np.zeros(len(epochs_a_filtered)), 
                        np.ones(len(epochs_b_filtered))])
    X_flattened = X.reshape(X.shape[0], -1)

    if sum(sum(X_train)) == 0:
        scores = 0
        ch_scores[channel] = scores
    else:
        scores = XGB_classification(X_flattened, y)
        ch_scores[channel] = scores

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "scores", "XGB.pickle")
with open(filepath, 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(ch_scores, file)
#   %% RNN classifier
ch_scores = {}
for channel in ch_names:
    # Concatenate data from two conditions and extract features
    X = np.concatenate([epochs_a_filtered.get_data(picks=channel),
                        epochs_b_filtered.get_data(picks=channel)])
    y = np.concatenate([np.zeros(len(epochs_a_filtered)),
                        np.ones(len(epochs_b_filtered))])
    y_binary = to_categorical(y)  # Convert labels to binary format for softmax
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary,
                                                        test_size=0.2,
                                                        random_state=42)
    
    if sum(sum(sum(X_train))) == 0:
        scores = 0
        ch_scores[channel] = scores
    else:
        scores = RNN_classification(X_train, X_test, y_train, y_test)
        ch_scores[channel] = scores

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "scores", "RNN.pickle")
with open(filepath, 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(ch_scores, file)



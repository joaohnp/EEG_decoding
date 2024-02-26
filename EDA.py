#%%
import matplotlib.pyplot as plt
import numpy as np

from mne import combine_evoked
from mne.datasets.limo import load_data
from mne.stats import linear_regression
from mne.viz import plot_compare_evokeds, plot_events
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import mne
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
import numpy as np
import numpy as np
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import os
#%% Only one subject in this dataset
subj = 1
limo_epochs = load_data(subject=subj)
#%% Plotting some basic information about the dataset
print(limo_epochs)
fig = plot_events(limo_epochs.events, event_id=limo_epochs.event_id)
fig.suptitle("Distribution of events in LIMO epochs")
print(limo_epochs.metadata.head())
# We want include all columns in the summary table
epochs_summary = limo_epochs.metadata.describe(include="all").round(3)
print(epochs_summary)
# only show -250 to 500 ms
ts_args = dict(xlim=(-0.25, 0.5))
times_ofinterest = [0.09, 0.15, 0.23]
# plot evoked response for face A
limo_epochs["Face/A"].average().plot_joint(
    times=times_ofinterest, title="Evoked response: Face A", ts_args=ts_args
)
# and face B
limo_epochs["Face/B"].average().plot_joint(
    times=times_ofinterest, title="Evoked response: Face B", ts_args=ts_args
)
#%% Getting basic info
ch_names = limo_epochs.info['ch_names']
#%% Plotting all plots from the same electrode
limo_epochs["Face/A"].plot_image(picks='A20', combine='mean')
#%%
epochs_a = limo_epochs["Face/A"]
epochs_b = limo_epochs["Face/B"]
X = np.concatenate([epochs_a.get_data(picks='A1'), epochs_b.get_data(picks='A1')])  # Feature matrix
y = np.concatenate([np.zeros(len(epochs_a)), np.ones(len(epochs_b))])  # Labels

#%% LDA_all channels
from LDA_classifier import LDA_classification
ch_scores = {}
for channel in ch_names:
    X = np.concatenate([epochs_a.get_data(picks=channel), epochs_b.get_data(picks=channel)])  # Feature matrix
    y = np.concatenate([np.zeros(len(epochs_a)), np.ones(len(epochs_b))])  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if sum(sum(sum(X_train))) == 0:
        scores=0
        ch_scores[channel] = scores
    else:
        scores = LDA_classification(X_train, y_train, X_test, y_test)
        ch_scores[channel] = scores

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "scores", "LDA.pickle")
with open(filepath, 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(ch_scores, file)
#%% SVM all channels
from SVM_classifier import SVM_classification
ch_scores = {}
for channel in ch_names:
    # Concatenate data from two conditions and extract features
    X = np.concatenate([epochs_a.get_data(picks=channel), epochs_b.get_data(picks=channel)])
    y = np.concatenate([np.zeros(len(epochs_a)), np.ones(len(epochs_b))])
    X = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if sum(sum(X_train)) == 0:
        scores=0
        ch_scores[channel] = scores
    else:
        scores = SVM_classification(X_train, y_train, X_test, y_test)
        ch_scores[channel] = scores

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "scores", "SVM.pickle")
with open(filepath, 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(ch_scores, file)

#%%
from XGBoost_classifier import XGB_classification
ch_scores = {}
for channel in ch_names:
    # Concatenate data from two conditions and extract features
    X = np.concatenate([epochs_a.get_data(picks=channel), epochs_b.get_data(picks=channel)])
    y = np.concatenate([np.zeros(len(epochs_a)), np.ones(len(epochs_b))])
    X_flattened = X.reshape(X.shape[0], -1)

    if sum(sum(X_train)) == 0:
        scores=0
        ch_scores[channel] = scores
    else:
        scores = XGB_classification(X_flattened, y)
        ch_scores[channel] = scores

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "scores", "XGB.pickle")
with open(filepath, 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(ch_scores, file)

#%%
from tensorflow.keras.utils import to_categorical
from RNN_classifier import RNN_classification

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from sklearn.model_selection import cross_val_score

def RNN_classification(X_train, X_test, y_train, y_test):
    # Build RNN model
    print(X_train.shape, X_test.shape,y_train.shape, y_test.shape)
    model = Sequential([
        LSTM(25, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2, verbose=1)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0) 
    print(f'RNN Accuracy: {accuracy:.2f}')

    return accuracy

ch_scores = {}
for channel in ch_names:
    # Concatenate data from two conditions and extract features
    X = np.concatenate([epochs_a.get_data(picks=channel), epochs_b.get_data(picks=channel)])
    y = np.concatenate([np.zeros(len(epochs_a)), np.ones(len(epochs_b))])
    y_binary = to_categorical(y)  # Convert labels to binary format for softmax
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    
    if sum(sum(sum(X_train))) == 0:
        scores=0
        ch_scores[channel] = scores
    else:
        scores = RNN_classification(X_train, X_test, y_train, y_test)
        ch_scores[channel] = scores

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "scores", "RNN.pickle")
with open(filepath, 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(ch_scores, file)

#%%
from CNN_classifier import CNN_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Assuming X is shaped as (samples, time steps, features) and y is categorical
y_binary = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Build CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

ch_scores = {}
for channel in ch_names:
    # Concatenate data from two conditions and extract features
    X = np.concatenate([epochs_a.get_data(picks=channel), epochs_b.get_data(picks=channel)])
    y = np.concatenate([np.zeros(len(epochs_a)), np.ones(len(epochs_b))])
    # Assuming X is shaped as (samples, time steps, features) and y is categorical
    y_binary = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    if sum(sum(sum(X_train))) == 0:
        scores=0
        ch_scores[channel] = scores
    else:
        scores = CNN_classification(X_train, X_test, y_train, y_test)
        ch_scores[channel] = scores

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "scores", "CNN.pickle")
with open(filepath, 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(ch_scores, file)
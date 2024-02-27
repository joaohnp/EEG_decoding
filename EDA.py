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
from sklearn.svm import SVC  # Support Vector Classifier
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from xgboost import XGBClassifier

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
all_ch_names = limo_epochs.info['ch_names']
bad_ch_names = limo_epochs.info['bads']
ch_names = [ch for ch in all_ch_names if ch not in bad_ch_names]


#%% Plotting all plots from the same electrode
limo_epochs["Face/A"].plot_image(picks='A20', combine='mean')
#%%
epochs_a = limo_epochs["Face/A"].crop(tmin=0, tmax=0.5)
epochs_b = limo_epochs["Face/B"].crop(tmin=0, tmax=0.5)
#%%
phase_coh = limo_epochs.metadata["phase-coherence"]
# get levels of phase coherence
levels = sorted(phase_coh.unique())
# create labels for levels of phase coherence (i.e., 0 - 85%)
labels = [f"{i:.2f}" for i in np.arange(0.0, 0.90, 0.05)]
labels_levels_dict = {label: level for label, level in zip(labels, levels)}

epochs_a_filtered = []
epochs_b_filtered = []
start = 0.50
end = 0.85

# Convert these labels to levels
level_range_start = labels_levels_dict[f"{start:.2f}"]
level_range_end = labels_levels_dict[f"{end:.2f}"]

# Filter the dataset
# Assuming your dataset uses 'phase-coherence' as a key for levels
epochs_a_filtered = epochs_a[epochs_a.metadata['phase-coherence'].between(level_range_start, level_range_end)]
epochs_b_filtered = epochs_b[epochs_b.metadata['phase-coherence'].between(level_range_start, level_range_end)]




#%%



# each trial has 201 elements. 
X = np.concatenate([epochs_a.get_data(picks='A1'), epochs_b.get_data(picks='A1')])  # Feature matrix
y = np.concatenate([np.zeros(len(epochs_a)), np.ones(len(epochs_b))])  # Labels

#%% LDA_all channels
from LDA_classifier import LDA_classification

ch_scores = {}
for channel in ch_names:
    X = np.concatenate([epochs_a_filtered.get_data(picks=channel),
                        epochs_b_filtered.get_data(picks=channel)])  # Feature matrix
    y = np.concatenate([np.zeros(len(epochs_a_filtered)), np.ones(len(epochs_b_filtered))])  # Labels
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
    X = np.concatenate([epochs_a_filtered.get_data(picks=channel),
                        epochs_b_filtered.get_data(picks=channel)])

    y = np.concatenate([np.zeros(len(epochs_a_filtered)),
                        np.ones(len(epochs_b_filtered))])
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
    X = np.concatenate([epochs_a_filtered.get_data(picks=channel), 
                        epochs_b_filtered.get_data(picks=channel)])
    y = np.concatenate([np.zeros(len(epochs_a_filtered)), 
                        np.ones(len(epochs_b_filtered))])
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
import numpy as np
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

#%%
from tensorflow.keras.utils import to_categorical

from RNN_classifier import RNN_classification


def RNN_classification(X_train, X_test, y_train, y_test):
    # Build RNN model
    # print(X_train.shape, X_test.shape,y_train.shape, y_test.shape)
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
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
    X = np.concatenate([epochs_a_filtered.get_data(picks=channel), 
                        epochs_b_filtered.get_data(picks=channel)])
    y = np.concatenate([np.zeros(len(epochs_a_filtered)), 
                        np.ones(len(epochs_b_filtered))])
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
# from CNN_classifier import CNN_classification
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.layers.experimental.preprocessing import Normalization

# Build CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def CNN_classification(X_train, X_test, y_train, y_test):
    input_shape = (1, 126)  # Adjust based on your actual data

    # normalizer = Normalization(axis=-1)  # You might need to adjust the axis depending on your data's shape
    # normalizer.adapt(X_train)
    model = Sequential([Normalization(input_shape=input_shape),
    Conv1D(filters=10, kernel_size=2, activation='relu', padding='same',
            input_shape=(1, 126)),
    MaxPooling1D(pool_size=2, padding='same'),
    Flatten(),
    Dense(40, activation='relu'),
    Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'CNN Accuracy: {accuracy:.2f}')


    return accuracy

# Assuming X is your input data with shape (samples, time steps, features)
ch_scores = {}
for channel in ch_names:
    # Concatenate data from two conditions and extract features
    X = np.concatenate([epochs_a_filtered.get_data(picks=channel),
                        epochs_b_filtered.get_data(picks=channel)])
    y = np.concatenate([np.zeros(len(epochs_a_filtered)), 
                        np.ones(len(epochs_b_filtered))])
    n_samples, n_time_steps, n_features = X.shape
    
    y_binary = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary,
                                                        test_size=0.2,
                                                        random_state=42)

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
# %%

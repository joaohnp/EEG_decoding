"""Module to check the outputs and find what's the best model to discern faces
"""
#%%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

#   %% Loading all scores
dir_path = os.path.dirname(os.path.realpath(__file__))
scores_path = os.path.join(dir_path, "scores")
all_scores = os.listdir(scores_path)
#%%

labels = ["LDA", "SVM", "XGB", "RNN"]

# Initialize an empty dictionary to hold the scores
scores_dict = {}

# Iterate through each label and its corresponding score file path
for label, score_path in zip(labels, all_scores):
    # Open the pickle file and load the scores
    with open(os.path.join(scores_path, score_path), "rb") as input_file:
        scores = pickle.load(input_file)
    # Assign the loaded scores to the corresponding label in the dictionary
    scores_dict[label] = scores

# %% Plotting

# Assuming scores_dict is already defined and contains scores for "LDA", "SVM", "XGB", "RNN"
ch_names = list(scores_dict["LDA"].keys())

# Initialize a dictionary to hold scores for plotting
channel_scores = {channel: [] for channel in ch_names}
threshold = 0.5
# Populate the dictionary with scores for each classifier
best_lda = 0
best_svm = 0
best_xgb = 0
best_rnn = 0

for channel in ch_names:
    LDA_score = np.mean(scores_dict["LDA"][channel])
    if LDA_score > best_lda:
        best_lda = LDA_score
    SVM_score = np.mean(scores_dict["SVM"][channel])
    if SVM_score > best_svm:
        best_svm = SVM_score
    XGB_score = np.mean(scores_dict["XGB"][channel])
    if XGB_score > best_xgb:
        best_xgb = XGB_score
    RNN_score=np.mean(scores_dict["RNN"][channel])
    if RNN_score > best_rnn:
        best_rnn = RNN_score

    if LDA_score > threshold:
        channel_scores[channel].append(LDA_score)
    else:
        channel_scores[channel].append(0)

    if SVM_score > threshold:
        channel_scores[channel].append(SVM_score)
    else:
        channel_scores[channel].append(0)
    if XGB_score > threshold:
        channel_scores[channel].append(XGB_score)
    else:
        channel_scores[channel].append(0)
    if RNN_score > threshold:
        channel_scores[channel].append(RNN_score)
    else:
        channel_scores[channel].append(0)

# Set up the plot
fig, ax = plt.subplots(figsize=(20, 6))

# Define the width of the bars and the offset
bar_width = 0.7
# offset = 0.1

# Labels for each classifier
labels = ["LDA", "SVM", "XGB", "RNN"]
colors = ['coral', 'darkgreen', 'navy', 'grey']

# Calculate positions for each group on the x-axis
x_positions = np.arange(len(ch_names))

# Plotting
for i, (label, color) in enumerate(zip(labels, colors)):
    # Calculate the offset position for each classifier's bars
    # positions = x_positions + (i - len(labels) / 2) * offset
    scores = [channel_scores[channel][i] for channel in ch_names]
    plt.bar(x_positions, scores, color=color, width=bar_width,
            label=label, align='center', alpha=1)

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Channel')
ax.set_ylabel('Scores')
ax.set_title('Scores by channel and classifier')
ax.legend()

# Set the position of the x ticks
plt.xticks(x_positions, ch_names, rotation=45)
# plt.tight_layout()
plt.ylim([0.475, 0.67])
plt.axhline(y = 0.5, color = 'r', linestyle = '--', alpha=0.5)
plt.show()


# %%

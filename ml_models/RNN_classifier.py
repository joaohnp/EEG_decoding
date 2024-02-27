"""Module to apply a simple RNN to classifiy ERPs per condition

    Returns
    -------
    accuracy
        returns the model effectivness in distinguishing the condition
    """
import numpy as np
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def RNN_classification(X_train, X_test, y_train, y_test):
    """Applies RNN to the fed in data

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

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 
                                                      X_train.shape[2])),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'RNN Accuracy: {accuracy:.2f}')

    return accuracy

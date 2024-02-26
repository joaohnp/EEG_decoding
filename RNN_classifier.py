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


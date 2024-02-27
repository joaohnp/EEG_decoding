import numpy as np
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def RNN_classification(X_train, X_test, y_train, y_test):

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X
                                                      _train.shape[2])),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2, verbose=1)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0) 
    print(f'RNN Accuracy: {accuracy:.2f}')

    return accuracy

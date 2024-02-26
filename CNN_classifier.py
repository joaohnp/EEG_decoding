import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

from sklearn.model_selection import cross_val_score

def CNN_classification(X_train, X_test, y_train, y_test):
    model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(1, 201)),
    MaxPooling1D(pool_size=2, padding='same'),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'CNN Accuracy: {accuracy:.2f}')


    return accuracy


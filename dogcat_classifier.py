import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import os
from keras.preprocessing import image

base_path = "C:/Users/HP/Documents/Image-Classifier"

X_train = np.loadtxt(os.path.join(base_path, 'input.csv'), delimiter=',')
Y_train = np.loadtxt(os.path.join(base_path, 'labels.csv'), delimiter=',')

X_test = np.loadtxt(os.path.join(base_path, 'input_test.csv'), delimiter=',')
Y_test = np.loadtxt(os.path.join(base_path, 'labels_test.csv'), delimiter=',')

X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)

X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=12, batch_size=64)

model.save(os.path.join(base_path, 'catvsdogmodel.h5'))
print("Model saved successfully!")

import sys
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scratchlayers import Conv2DScratch, SimpleRNNScratch, LSTMScratch

def create_cnn():
    model = keras.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1)),
        Conv2DScratch(filters=32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    return model

def create_rnn():
    model = keras.Sequential([
        layers.InputLayer(input_shape=(100, 50)),
        SimpleRNNScratch(units=64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def create_lstm():
    model = keras.Sequential([
        layers.InputLayer(input_shape=(100, 50)),
        LSTMScratch(units=64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model
    

# Compile dan display model
if __name__ == "__main__":
    model = create_cnn()
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Example: Load MNIST data lalu train
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]
    
    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=5,
        validation_data=(x_test, y_test),
        verbose=1
    )

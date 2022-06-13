import tensorflow
from keras import Sequential
from keras.layers import CuDNNLSTM, ELU, Bidirectional, LSTM

from keras.layers import (
    Dense, Dropout
)


def get_model(shape, n_classes: int):
    model = tensorflow.keras.Sequential()
    model.add(LSTM(128, input_shape=shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))


    return model

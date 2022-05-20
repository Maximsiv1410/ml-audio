import tensorflow
from tensorflow.python.keras.layers import LSTM, Dropout, Dense


def get_model(shape, n_classes: int):
    model = tensorflow.keras.Sequential()

    #model.add(GRU(256, input_shape=shape))
    model.add(LSTM(256, input_shape=shape))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation='softmax'))

    return model
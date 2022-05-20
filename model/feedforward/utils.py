from keras import Sequential
from keras.layers import (
    Dense, Dropout
)


def get_model(shape_length, n_classes: int):
    model = Sequential()
    #
    # model.add(Dense(512, activation='relu', input_shape=(shape_length,)))
    # model.add(Dropout(0.5))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(n_classes, activation='softmax'))

    model.add(Dense(256, activation="relu", input_shape=(shape_length,)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    # output layer
    model.add(Dense(n_classes, activation="softmax"))

    return model

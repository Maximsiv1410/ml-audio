from dataclasses import dataclass

from keras import Sequential
from keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    SpatialDropout2D,
    LeakyReLU,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D
)
from keras.regularizers import l2

import tensorflow as tf

from tensorflow.python.keras import layers, models



import matplotlib.pyplot as plt
@dataclass
class Shape:
    rows: int
    frames: int
    channels: int = 1

    def to_tuple(self):
        return self.rows, self.frames, self.channels


@dataclass
class CNNConfig:
    shape: Shape
    classes_count: int
    spatial_dropout_first_level: float
    spatial_dropout_second_level: float
    l2_rate: float

def safe(config: CNNConfig):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=config.shape.to_tuple()))
    model.add(layers.SpatialDropout2D(0.07))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.07))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.15))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(config.classes_count, activation='softmax'))

    return model



def get_model(config: CNNConfig):
    model = models.Sequential()

    kernel_size = (3, 3)

    model.add(layers.Conv2D(32, kernel_size, activation='relu', input_shape=config.shape.to_tuple()))
    model.add(BatchNormalization())

    model.add(layers.SpatialDropout2D(0.07))
    model.add(layers.Conv2D(32, kernel_size, activation='relu'))
    model.add(BatchNormalization())

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.SpatialDropout2D(0.07))
    model.add(layers.Conv2D(64, kernel_size, activation='relu'))
    model.add(BatchNormalization())

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.SpatialDropout2D(0.15))
    model.add(layers.Conv2D(64, kernel_size, activation='relu'))
    model.add(BatchNormalization())

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(config.classes_count, activation='softmax'))



    return model

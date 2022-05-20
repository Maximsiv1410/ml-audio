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


def get_model(config: CNNConfig):
    """ too heavy model, needs to be refined """
    model = Sequential()

    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     kernel_regularizer=l2(config.l2_rate),
                     input_shape=config.shape.to_tuple()))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(config.spatial_dropout_first_level))

    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     kernel_regularizer=l2(config.l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SpatialDropout2D(config.spatial_dropout_first_level))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     kernel_regularizer=l2(config.l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(config.spatial_dropout_second_level))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     kernel_regularizer=l2(config.l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling2D())

    model.add(Dense(config.classes_count, activation='softmax'))

    return model

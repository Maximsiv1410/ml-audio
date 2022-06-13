from collections import defaultdict
from datetime import datetime

import visualkeras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

import os
import json
import numpy as np

from model.lstm_utils import *
from core.report import make_report


def load_classes(path):
    with open(path, 'r') as file:
        return json.load(file)


def load_data(data_path):
    features = np.load(f'{data_path}/x.npy', allow_pickle=True).astype('float32')
    labels = np.load(f'{data_path}/y.npy', allow_pickle=True).astype('str')
    classes = load_classes(f'{data_path}/class_map.json')

    print(len(labels))

    return features, labels, classes

def fit_lstm(data_path, epochs=100, batch_size=32, model_path='lstm.hdf5'):
    features, labels, classes = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    print(X_train.shape[1:])
    print(y_test.shape)

    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    y_train_encoded = le.fit_transform(y_train)

    model = None
    if os.path.exists(model_path):
        print('load from file')
        model = keras.models.load_model(model_path)
    else:
        print('build from scratch')
        model = get_model(X_train.shape[1:], len(classes))

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
            optimizer=keras.optimizers.Adam(1e-4))

        #model.build(features.shape[1:])

    color_map = defaultdict(dict)
    color_map[Dense]['fill'] = 'green'
    color_map[LSTM]['fill'] = 'red'

    visualkeras.layered_view(model, to_file='lstm_vis.png',legend=True)

    print(model.summary())

    start = datetime.now()
    history = model.fit(X_train,
                        y_train_encoded,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=1 / 12.,
                        verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    #model.save(model_path)
    make_report('lstm', model, history, classes, X_train, y_train_encoded, X_test, y_test_encoded)


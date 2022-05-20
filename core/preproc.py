import json
import os

import librosa as librosa
import numpy as np

from core.audio import extract_feature


def prepare_dataset(
        dataset_path,
        result_path,
        interests,
        target_sr=44100,
        target_duration=1,
        n_mels=64
):
    features = []
    classes = []
    labels = []

    for entry in os.scandir(dataset_path):
        if entry.is_dir() and entry.name in interests:
            print(entry.name)
            classes.append(entry.name)
            for subentry in os.scandir(entry):
                if subentry.is_file():
                    y, sr = librosa.load(subentry)
                    feature = extract_feature(y, sr, target_sr, target_duration, n_mels)

                    features.append(feature)
                    labels.append(entry.name)   # current class id

    np.save(f'{result_path}/x.npy', np.array(features, dtype=object))
    np.save(f'{result_path}/y.npy', np.array(labels, dtype=object))

    mapping = {class_id: class_name for class_id, class_name in enumerate(classes)}
    with open(f'{result_path}/class_map.json', 'w') as file:
        json.dump(mapping, file)

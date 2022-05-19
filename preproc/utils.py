import json
from datetime import datetime
import os

import librosa as librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras


def truncate(y, sr, target_duration):
    # sample_rate * target_duration is just a number of samples we need to take
    n_samples = int(sr * target_duration)
    return y[:n_samples]

def resize_audio(y, sr, target_duration):
    actual_duration = librosa.get_duration(y=y, sr=sr)
    # just len(y) / sr btw
    if actual_duration < target_duration:
        required_size = target_duration * sr
        return librosa.util.pad_center(y, size=required_size, mode='constant')
    else:
        # if actual_duration is greater or equal -> just truncate it
        return truncate(y, sr, target_duration)


def extract_feature(
        y,
        origin_sr: int,
        target_sr: int,
        target_duration: int,
        n_mels: int
):
    # resample
    resampled = librosa.resample(y=y, orig_sr=origin_sr, target_sr=target_sr)
    #

    # pad
    resized = resize_audio(y=resampled, sr=target_sr, target_duration=target_duration)

    # time shift
    #

    # get mel
    # normalized_y = librosa.util.normalize(resized)
    mel = librosa.feature.melspectrogram(y=resized, sr=target_sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(abs(mel))
    normalized_mel = librosa.util.normalize(mel_db)

    return normalized_mel


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
            classes.append(entry.name)
            for subentry in os.scandir(entry):
                if subentry.is_file():
                    y, sr = librosa.load(subentry)
                    feature = extract_feature(y, sr, target_sr, target_duration, n_mels)

                    features.append(feature)
                    labels.append(len(classes)-1)   # current class id

    np.save(f'{result_path}/x.npy', np.array(features, dtype=object))
    np.save(f'{result_path}/y.npy', np.array(labels, dtype=object))

    mapping = {class_id: class_name for class_id, class_name in enumerate(classes)}
    with open(f'{result_path}/class_map.json', 'w') as file:
        json.dump(mapping, file)

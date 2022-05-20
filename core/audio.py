import librosa as librosa

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

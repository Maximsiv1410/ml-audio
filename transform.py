import click


from core.preproc import prepare_dataset
from core.system import get_config


def do_transform(config_path):
    config = get_config(config_path)["General"]
    prepare_dataset(config['dataset_path'],
                    config['result_path'],
                    config['classes'],
                    int(config['target_sr']),
                    int(config['target_duration']),
                    int(config['mels']))


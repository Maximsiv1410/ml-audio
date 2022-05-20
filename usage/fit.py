from usage import factory

from core.system import get_config

def do_fit(config_path):
    config = get_config(config_path)['General']

    arch = config['arch']
    epochs = int(config['epochs'])
    batch_size = int(config['batch_size'])
    dataset_path = config['dataset_path']

    if arch not in ['fnn', 'cnn', 'rnn']:
        raise Exception(f'Unknown architecture: {arch}')

    factory_method = getattr(factory, f"fit_{arch}")
    return factory_method(dataset_path, epochs, batch_size)


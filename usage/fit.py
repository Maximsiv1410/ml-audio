from usage import factory

from core.system import get_config


def do_fit(config_path):
    config = get_config(config_path)['Fit']

    arch = config['arch']
    epochs = int(config['epochs'])
    batch_size = int(config['batch_size'])
    cooked_dataset = config['cooked_dataset']

    if arch not in ['mlp', 'cnn', 'lstm']:
        raise Exception(f'Unknown architecture: {arch}')

    factory_method = getattr(factory, f"fit_{arch}")
    return factory_method(cooked_dataset, epochs, batch_size)


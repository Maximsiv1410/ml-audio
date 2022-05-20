import factory

def do_predict(config):
    arch = ''

    if arch not in ['fnn', 'cnn', 'rnn']:
        raise Exception()

    factory_method = getattr(factory, f"predict_{arch}")
    return factory_method(config)
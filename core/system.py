import configparser

def get_config(path):
    config = configparser.ConfigParser()
    with open(path, 'r') as file:
        config.read_file(file)
        return config
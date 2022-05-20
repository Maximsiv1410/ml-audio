import click
from usage import factory


@click.command()
@click.option("--action",
              help="Type of action to do")
@click.option("--config",
              type=click.Path(exists=True),
              help="Path to the config.")
def main(action, config):
    if action not in ['fit', 'predict', 'transform']:
        raise Exception(f'Unknown action: {action}')

    factory_method = getattr(factory, f"do_{action}")
    return factory_method(config)


if __name__ == '__main__':
    main()

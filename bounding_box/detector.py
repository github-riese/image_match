import pandas as pd


def learn_boxes(args: list):
    config = configure(args)
    x = pd.read_csv(config['csv_in'])
    print(x)


def configure(args: list) -> dict:
    return {
        'csv_in': args[1]
    }

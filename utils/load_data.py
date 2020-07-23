import pandas as pd

from config.config import Config


def load_data():
    data = pd.read_csv(Config.data_path).set_index(Config.data_index)
    return data

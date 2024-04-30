import pandas as pd


def read_raw_data(train: bool = False, valid: bool = False, test: bool = False) -> pd.DataFrame:
    """
    Load raw data corresponding to the train, valid and/or test sets.
    """

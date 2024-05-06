import numpy as np
import pandas as pd


def rebalance_classes(df_train: pd.DataFrame, response_col: str) -> pd.DataFrame:
    """
    Resample the class with lowest representation
    """
    single_prop = sum(df_train[response_col] == 0) / len(df_train)
    class_to_oversample = 1 if single_prop > 0.5 else 0
    oversample_rate = 0.5 - min(single_prop, 0.5 - single_prop)
    oversample_n = int(np.floor(oversample_rate * len(df_train)))
    np.random.seed(0)
    new_indices = np.random.choice(df_train.loc[df_train[response_col] == class_to_oversample, :].index, oversample_n)
    new_df = df_train.loc[new_indices, :]
    return pd.concat([df_train, new_df], axis=0, ignore_index=True)


def add_fake_features(df_train, n_fake_features) -> tuple[pd.DataFrame, list[str]]:
    """
    Add noise variables to help guage overfitting
    """
    fake_features = []
    for i in range(n_fake_features):
        fake_feature_col = f"fake_{i}"
        df_train[fake_feature_col] = np.random.randn(df_train.shape[0])
        fake_features.append(fake_feature_col)
    return df_train, fake_features


def post_warmup_locator(df: pd.DataFrame, minimum_observations: int, minimum_elapsed_time_hours: float):
    """
    It may take several observations before a rate (count variable divided by time) is stable enough to be used for predictions.
    Returns the indices of the dataset that have passed this warmup period.
    """
    pass_minimum_observations = df["total_all_locations_cumulative"] > minimum_observations
    pass_minimum_elapsed_time = df["elapsed_time_hours"] > minimum_elapsed_time_hours
    return pass_minimum_observations & pass_minimum_elapsed_time

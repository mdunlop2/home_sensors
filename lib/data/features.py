import datetime as dt
import sqlite3

import numpy as np
import pandas as pd

RAW_SCHEMA = {
    "multiple_occupancy": "int64",
    "home_id": "str",
    "id": "str",
    "location": "str",
    "datetime": "datetime64[ns, UTC]",
}


def read_raw_data(database_location: str, train: bool = False, valid: bool = False, test: bool = False) -> pd.DataFrame:
    """
    Load raw data corresponding to the train, valid and/or test sets.
    """
    conn = sqlite3.connect(database_location)
    set_conditions = []
    if train:
        set_conditions.append("is_train")
    if valid:
        set_conditions.append("is_valid")
    if test:
        set_conditions.append("(not is_train) and (not is_valid)")
    if len(set_conditions) == 0:
        raise ValueError("At least one of train, valid or test sets should be specified!")

    set_condition = "where" + "OR".join(f"({condition})" for condition in set_conditions)
    sql = f"""
    select homes.id as home_id, homes.multiple_occupancy, motion.id, motion.datetime, motion.location from homes
    inner join motion
    on homes.id = motion.home_id
    inner join (
        select home_id from train_valid_test
        {set_condition}
    ) tvt
    on homes.id = tvt.home_id
    """
    return pd.read_sql(sql, conn, dtype=RAW_SCHEMA)


def transform_sensor_triggers_to_time_series(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a sequence of sensor triggers into a time series where each column represents a trigger in a location.
    """
    time_series = (
        raw_data.assign(ones=1)
        .pivot_table(index=["home_id", "datetime"], columns=["location"], values="ones", aggfunc="first")
        .fillna(0)
        .sort_index()
        .astype(np.int64)
    )
    time_series.columns.name = None
    return time_series.reset_index()


def _multiple_columns_1_in_window(x):
    sum_per_column = np.sum(x, axis=0)
    columns_with_at_least_1_motion = np.sum(sum_per_column > 0)
    return columns_with_at_least_1_motion > 1


def _add_multiple_room_triggers_in_window_single_home(
    time_series: pd.DataFrame, window: str, locations: list[str]
) -> pd.DataFrame:
    wide_result = (
        time_series[["datetime"] + locations]
        .rolling(window, on="datetime", method="table")
        .apply(_multiple_columns_1_in_window, engine="numba", raw=True)
    )
    time_series[f"multiple_room_triggers_{window}"] = wide_result[locations[0]].astype(
        int
    )  # can take any location as they all have same values
    return time_series


def add_multiple_location_triggers_in_window(
    time_series: pd.DataFrame, window: str, locations: list[str]
) -> pd.DataFrame:
    """
    1 if multiple rooms were triggered during the time window ending at that minute, 0 otherwise
    """
    results = []
    for _, df in time_series.groupby("home_id"):
        results.append(_add_multiple_room_triggers_in_window_single_home(df, window, locations))
    time_series = pd.concat(results, axis=0, ignore_index=True)
    return time_series.sort_values(["home_id", "datetime"])


def add_cumulative_triggers(time_series: pd.DataFrame, locations: list[str]) -> pd.DataFrame:
    """
    Record cumulative sensor counts per location
    """
    time_series["total"] = time_series[locations].sum(axis=1)
    cumulative = time_series.groupby("home_id")[locations + ["total"]].cumsum()
    cumulative.columns = [col + "_cumulative" for col in cumulative.columns]
    return pd.concat([time_series, cumulative], axis=1)


def add_elapsed_time(time_series: pd.DataFrame) -> pd.DataFrame:
    """
    Add the cumulative time that has passed since the first sensor trigger at each home
    """
    start_time = (
        time_series.groupby(["home_id"], as_index=False)
        .agg({"datetime": "min"})
        .rename(columns={"datetime": "start_datetime"})
    )
    time_series = pd.merge(time_series, start_time, on="home_id")
    time_series["elapsed_time_hours"] = (time_series["datetime"] - time_series["start_datetime"]) / dt.timedelta(
        hours=1
    )
    return time_series

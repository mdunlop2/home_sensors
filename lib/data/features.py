import sqlite3

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

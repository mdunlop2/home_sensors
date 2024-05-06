import sqlite3

import numpy as np
import pandas as pd

from lib.common.logging import get_logger
from lib.common.tables import HOMES_TABLE, MOTION_TABLE, TRAIN_VALID_TEST_TABLE, table_has_data

_LOGGER = get_logger(__name__)

TRAIN_PROP = 0.5
VALID_PROP = 0.25
TEST_PROP = 1.0 - TRAIN_PROP - VALID_PROP
SPLIT_SEED = 0


def add_train_valid_test_split_table(database_location: str) -> None:
    """
    Add train-valid-test indicators to the database
    """
    conn = sqlite3.connect(database_location)
    if table_has_data(database_location, TRAIN_VALID_TEST_TABLE):
        _LOGGER.info(f"Table {TRAIN_VALID_TEST_TABLE} already exists, not adding again.")
        return
    all_home_id_sql = f"""
    select distinct(home_id) from {MOTION_TABLE}
    where home_id in (select id from {HOMES_TABLE})
    """
    homes = pd.read_sql(all_home_id_sql, conn)
    n_homes = len(homes)
    np.random.seed(SPLIT_SEED)
    idx = np.arange(n_homes)
    random_order = idx.copy()
    np.random.shuffle(random_order)

    homes = homes.iloc[random_order, :]
    train_rows = np.ceil(TRAIN_PROP * n_homes).astype(np.int64)
    valid_rows = np.ceil(VALID_PROP * n_homes).astype(np.int64)
    homes["is_train"] = idx < train_rows
    homes["is_valid"] = (~homes["is_train"]) & (idx < train_rows + valid_rows)
    homes.to_sql(TRAIN_VALID_TEST_TABLE, conn, index=False, if_exists="replace")
    _LOGGER.info(f"Table {TRAIN_VALID_TEST_TABLE} added successfully.")

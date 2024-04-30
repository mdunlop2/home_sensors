import sqlite3

HOMES_TABLE = "homes"
MOTION_TABLE = "motion"
TRAIN_VALID_TEST_TABLE = "train_valid_test"


def table_has_data(database_location: str, table_name: str):
    """
    Check if there is at least one row from the table.
    """
    conn = sqlite3.connect(database_location)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    table_exists = cursor.fetchone()

    if table_exists:
        cursor.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
        row_exists = cursor.fetchone()
        return row_exists
    return False

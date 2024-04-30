import tempfile

from lib.data.raw import download_raw_data_if_not_exists
from lib.data.split import add_train_valid_test_split_table


def test_full_pipeline() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db") as temp_db_file:
        download_raw_data_if_not_exists(temp_db_file.name)
        add_train_valid_test_split_table(temp_db_file.name)

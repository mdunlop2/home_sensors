import tempfile

from lib.data.features import add_all_features, read_raw_data
from lib.data.raw import download_raw_data_if_not_exists
from lib.data.split import add_train_valid_test_split_table


def test_full_pipeline() -> None:
    """
    Run through the entire pipeline used in the notebook series to ensure it functions correctly
    """
    with tempfile.NamedTemporaryFile(suffix=".db") as temp_db_file:
        download_raw_data_if_not_exists(temp_db_file.name)
        add_train_valid_test_split_table(temp_db_file.name)

        # add features
        multi_location_windows = ["5min", "30min", "1h", "2h"]
        df = read_raw_data(temp_db_file.name, train=True)
        _ = add_all_features(df, multi_location_windows)

        # transformation

        # model fit

        # model validations

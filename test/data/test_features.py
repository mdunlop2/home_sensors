import datetime as dt

import pandas as pd

from lib.data.features import (
    add_cumulative_triggers,
    add_elapsed_time,
    add_multiple_location_triggers_in_window,
    transform_sensor_triggers_to_time_series,
)

_SAMPLE_LOCATIONS = ["bedroom1", "bathroom1", "hallway"]


def get_sample_raw_data() -> pd.DataFrame:
    """example raw data"""
    df_a = pd.DataFrame(
        {
            "multiple_occupancy": 1,
            "home_id": "a",
            "datetime": pd.to_datetime(
                [
                    dt.datetime(2024, 1, 1),
                    dt.datetime(2024, 1, 1, 1),
                    dt.datetime(2024, 1, 1, 1),
                    dt.datetime(2024, 1, 1, 2),
                ]
            ),
            "location": ["bedroom1", "bedroom1", "bathroom1", "bathroom1"],
        }
    )
    df_b = pd.DataFrame(
        {
            "multiple_occupancy": 0,
            "home_id": "b",
            "datetime": pd.to_datetime([dt.datetime(2024, 1, 1)]),
            "location": ["hallway"],
        }
    )
    return pd.concat([df_a, df_b], axis=0, ignore_index=True)


def get_sample_time_series() -> pd.DataFrame:
    """time series of sensor triggers"""
    df_a = pd.DataFrame(
        {
            "home_id": "a",
            "datetime": pd.to_datetime(
                [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1, 1), dt.datetime(2024, 1, 1, 2)]
            ),
            "multiple_occupancy": 1,
            "bathroom1": [0, 1, 1],
            "bedroom1": [1, 1, 0],
            "hallway": 0,
            "total_all_locations": [1, 2, 1],
        }
    )
    df_b = pd.DataFrame(
        {
            "home_id": "b",
            "datetime": pd.to_datetime([dt.datetime(2024, 1, 1)]),
            "multiple_occupancy": 0,
            "bathroom1": [0],
            "bedroom1": [0],
            "hallway": [1],
            "total_all_locations": [1],
        }
    )
    return pd.concat([df_a, df_b], axis=0, ignore_index=True)


def get_sample_time_series_with_2h_multiple_location_trigger():
    """sample time series with multiple locations triggered within a window"""
    time_series = get_sample_time_series()
    feature = pd.DataFrame(
        {
            "home_id": ["a", "a", "a", "b"],
            "datetime": pd.to_datetime(
                [
                    dt.datetime(2024, 1, 1),
                    dt.datetime(2024, 1, 1, 1),
                    dt.datetime(2024, 1, 1, 2),
                    dt.datetime(2024, 1, 1),
                ]
            ),
            "multiple_room_triggers_2h": [0, 1, 1, 0],
        }
    )
    return pd.merge(time_series, feature, on=["home_id", "datetime"])


def get_sample_time_series_with_cumulative_triggers():
    """sample time series with cumulative triggers"""
    time_series = get_sample_time_series()
    feature = pd.DataFrame(
        {
            "home_id": ["a", "a", "a", "b"],
            "datetime": pd.to_datetime(
                [
                    dt.datetime(2024, 1, 1),
                    dt.datetime(2024, 1, 1, 1),
                    dt.datetime(2024, 1, 1, 2),
                    dt.datetime(2024, 1, 1),
                ]
            ),
            "bedroom1_cumulative": [1, 2, 2, 0],
            "bathroom1_cumulative": [0, 1, 2, 0],
            "hallway_cumulative": [0, 0, 0, 1],
            "total_all_locations_cumulative": [1, 3, 4, 1],
        }
    )
    return pd.merge(time_series, feature, on=["home_id", "datetime"])


def get_sample_time_series_with_elapsed_time() -> pd.DataFrame:
    """sample time series with elapsed time from first sensor per home"""
    time_series = get_sample_time_series()
    feature = pd.DataFrame(
        {
            "home_id": ["a", "a", "a", "b"],
            "datetime": pd.to_datetime(
                [
                    dt.datetime(2024, 1, 1),
                    dt.datetime(2024, 1, 1, 1),
                    dt.datetime(2024, 1, 1, 2),
                    dt.datetime(2024, 1, 1),
                ]
            ),
            "start_datetime": pd.to_datetime(dt.datetime(2024, 1, 1)),
            "elapsed_time_hours": [0.0, 1.0, 2.0, 0.0],
        }
    )
    return pd.merge(time_series, feature, on=["home_id", "datetime"])


def test_transform_sensor_triggers_to_time_series() -> None:
    """verify transformation to a time series"""
    raw_data = get_sample_raw_data()
    expected_time_series = get_sample_time_series()
    time_series = transform_sensor_triggers_to_time_series(raw_data)
    pd.testing.assert_frame_equal(time_series, expected_time_series)


def test_multiple_location_triggers() -> None:
    """verify triggers in multiple locations within a window"""
    time_series = get_sample_time_series()
    window = "2h"
    expected_result = get_sample_time_series_with_2h_multiple_location_trigger()
    result = add_multiple_location_triggers_in_window(time_series, window, _SAMPLE_LOCATIONS)
    pd.testing.assert_frame_equal(result, expected_result)


def test_cumulative_triggers() -> None:
    """verify cumulative triggers perform as expected"""
    time_series = get_sample_time_series()
    expected_result = get_sample_time_series_with_cumulative_triggers()
    columns_to_sum = _SAMPLE_LOCATIONS + ["total_all_locations"]
    result = add_cumulative_triggers(time_series, columns_to_sum)
    pd.testing.assert_frame_equal(result, expected_result)


def test_elapsed_time() -> None:
    """verify elapsed time as expected"""
    time_series = get_sample_time_series()
    expected_result = get_sample_time_series_with_elapsed_time()
    result = add_elapsed_time(time_series)
    pd.testing.assert_frame_equal(result, expected_result)


if __name__ == "__main__":
    time_series = get_sample_time_series()
    expected_result = get_sample_time_series_with_cumulative_triggers()
    columns_to_sum = _SAMPLE_LOCATIONS + ["total_all_locations"]
    result = add_cumulative_triggers(time_series, columns_to_sum)
    pd.testing.assert_frame_equal(result, expected_result)

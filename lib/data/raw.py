import requests

from lib.common.logging import get_logger
from lib.common.tables import MOTION_TABLE, table_has_data

_LOGGER = get_logger(__name__)


_RAW_DATA_URL = "https://imperialcollegelondon.box.com/shared/static/8se12flvva8jcpqwehuyuxob0uj4kfr4.db"

RAW_SCHEMA = {
    "multiple_occupancy": "int64",
    "home_id": "str",
    "id": "str",
    "location": "str",
    "datetime": "datetime64[ns, UTC]",
}


def download_raw_data_if_not_exists(database_location) -> None:
    """
    Download the raw data to a location within the repository for ease of access.
    """
    if table_has_data(database_location, MOTION_TABLE):
        _LOGGER.info(f"Not downloading again, {database_location} already exists.")
    else:
        response = requests.get(_RAW_DATA_URL, timeout=2)
        if response.status_code == 200:
            with open(database_location, "wb") as f:
                f.write(response.content)
            _LOGGER.info(f"Raw data downloaded successfully to {database_location}")
        else:
            _LOGGER.info(f"Failed to download {_RAW_DATA_URL}. The response was:\n{repr(response)}")

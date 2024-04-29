import logging
import os

import requests

from lib.common.paths import DATA_RAW

_LOGGER = logging.getLogger(__name__)


_RAW_DATA_URL = "https://imperialcollegelondon.box.com/shared/static/8se12flvva8jcpqwehuyuxob0uj4kfr4.db"
_RAW_FILE_NAME = "data.db"
RAW_FILE_LOCATION = os.path.join(DATA_RAW, _RAW_FILE_NAME)

RAW_SCHEMA = {
    "multiple_occupancy": "int64",
    "home_id": "str",
    "id": "str",
    "location": "str",
    "datetime": "datetime64[ns, UTC]"
}


def download_raw_data_if_not_exists() -> None:
    """
    Download the raw data to a location within the repository for ease of access.
    """
    if os.path.exists(RAW_FILE_LOCATION):
        _LOGGER.debug(f"Not downloading again, {RAW_FILE_LOCATION} already exists.")
    else:
        response = requests.get(_RAW_DATA_URL, timeout=2)
        if response.status_code == 200:
            with open(RAW_FILE_LOCATION, "wb") as f:
                f.write(response.content)
            _LOGGER.info(f"Raw data downloaded successfully to {RAW_FILE_LOCATION}")
        else:
            _LOGGER.info(f"Failed to download {_RAW_DATA_URL}. The response was:\n{repr(response)}")

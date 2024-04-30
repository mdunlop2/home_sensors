"""
Navigation utilities to help with deployment in an arbitrary location.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
_DATABASE_NAME = "data.db"
DATABASE_LOCATION = os.path.join(DATA, _DATABASE_NAME)

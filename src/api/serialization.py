"""Convert API payloads to strict JSON-compatible Python values.

Non-finite numeric values (NaN and either infinity) use JSON ``null``. Missing
pandas values also use ``null``. Unsupported objects raise ``TypeError`` so a
data-contract problem is visible instead of being hidden by string conversion.
"""
from collections.abc import Mapping
from datetime import date, datetime, time
import math

import numpy as np
import pandas as pd


def to_json_compatible(value, *, path="$"):
    """Recursively convert NumPy/pandas values while retaining response fields."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    # Check pandas' explicit missing sentinels before date/scalar conversion.
    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, np.ndarray):
        return to_json_compatible(value.tolist(), path=path)

    if isinstance(value, np.generic):
        return to_json_compatible(value.item(), path=path)

    if isinstance(value, (np.datetime64,)):
        return to_json_compatible(value.item(), path=path)

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, pd.Period):
        return value.to_timestamp().isoformat()

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool, np.generic)):
                raise TypeError(f"Unsupported JSON object key at {path}: {type(key).__name__}")
            safe_key = to_json_compatible(key, path=f"{path}.<key>")
            if safe_key is None or isinstance(safe_key, (dict, list)):
                raise TypeError(f"Unsupported JSON object key at {path}: {type(key).__name__}")
            converted[safe_key] = to_json_compatible(item, path=f"{path}.{safe_key}")
        return converted

    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item, path=f"{path}[{index}]") for index, item in enumerate(value)]

    raise TypeError(f"Unsupported response value at {path}: {type(value).__name__}")

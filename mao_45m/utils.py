__all__ = ["sleep", "take", "to_datetime", "to_timedelta"]


# standard library
from collections.abc import Iterator
from contextlib import contextmanager
from logging import getLogger
from time import perf_counter, sleep as sleep_


# dependencies
import numpy as np
import pandas as pd
from dateparser import parse


# constants
LOGGER = getLogger(__name__)


def sleep(seconds: float, /) -> None:
    """Busy-waiting sleep (more precise but more CPU usage)."""
    start = perf_counter()
    end = start + seconds

    while perf_counter() < end:
        pass


@contextmanager
def take(duration: float, /, *, precise: bool = False) -> Iterator[None]:
    """Run a code block for a specified duration.

    Args:
        duration: Run time of the code block in seconds.
        precise: Whether to use busy-waiting sleep for more precise timing.

    """
    start = perf_counter()
    yield
    end = perf_counter()

    if (elapsed := end - start) > duration:
        LOGGER.warning(f"Block run exceeds {duration} s.")
    else:
        if precise:
            sleep(duration - elapsed)
        else:
            sleep_(duration - elapsed)


def to_datetime(value: np.datetime64 | str, /) -> np.datetime64:
    """Parse a string into a NumPy datetime64[ns] object in UTC."""
    if isinstance(value, np.datetime64):
        return value.astype("M8[ns]")

    if (parsed := parse(value)) is None:
        raise ValueError(f"Could not parse to datetime: {value!s}")

    return pd.to_datetime(parsed).tz_convert("UTC").to_datetime64()


def to_timedelta(value: np.timedelta64 | str | float, /) -> np.timedelta64:
    """Parse a string or float into a NumPy timedelta64[ns] object."""
    if isinstance(value, np.timedelta64):
        return value.astype("m8[ns]")

    if isinstance(value, str):
        return pd.to_timedelta(value).to_timedelta64()

    return pd.to_timedelta(value, "s").to_timedelta64()

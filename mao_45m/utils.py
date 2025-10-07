__all__ = ["log", "sleep", "take", "to_datetime", "to_timedelta"]


# standard library
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from logging import FileHandler, Formatter, LogRecord, StreamHandler, getLogger
from os import PathLike
from time import perf_counter, sleep as sleep_


# dependencies
import numpy as np
import pandas as pd
from dateparser import parse


# constants
LOGGER = getLogger(__name__)
LOGGER_FORMAT = "{asctime} | {name} | {levelname} | {message}"


class ISO8601Formatter(Formatter):
    """Formatter whose default time format is ISO 8601."""

    def formatTime(self, record: LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, timezone.utc).isoformat()


@contextmanager
def log(
    *,
    file: PathLike[str] | str | None = None,
    file_level: int | str = "DEBUG",
    stderr: bool = True,
    stderr_level: int | str = "INFO",
) -> Iterator[None]:
    """Context manager for the root logger configuration.

    Args:
        file: Path to the log file.
        file_level: Logging level for the log file.
        stderr: Whether to log to stderr.
        stderr_level: Logging level for stderr.

    """
    root = getLogger()
    current_level = root.level
    current_handlers = root.handlers.copy()
    formatter = ISO8601Formatter(LOGGER_FORMAT, style="{")

    try:
        root.setLevel("DEBUG")

        for handler in root.handlers:
            root.removeHandler(handler)

        if file is not None:
            handler = FileHandler(file)
            handler.setFormatter(formatter)
            handler.setLevel(file_level)
            root.addHandler(handler)

        if stderr:
            handler = StreamHandler()
            handler.setFormatter(formatter)
            handler.setLevel(stderr_level)
            root.addHandler(handler)

        yield
    finally:
        root.setLevel(current_level)

        for handler in root.handlers:
            root.removeHandler(handler)

        for handler in current_handlers:
            root.addHandler(handler)


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
        LOGGER.warning(f"Block run exceeded {duration} s.")
    else:
        LOGGER.debug(f"Block run finished in {elapsed} s.")

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

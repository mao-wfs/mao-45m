__all__ = ["sleep", "take"]


# standard library
from collections.abc import Iterator
from contextlib import contextmanager
from logging import getLogger
from time import perf_counter, sleep as sleep_


# constants
LOGGER = getLogger(__name__)


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


def sleep(seconds: float, /) -> None:
    """Busy-waiting sleep (more precise but more CPU usage)."""
    start = perf_counter()
    end = start + seconds

    while perf_counter() < end:
        pass

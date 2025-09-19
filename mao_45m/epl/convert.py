__all__ = ["get_feed"]


# standard library
from collections.abc import Sequence


# dependencies
import numpy as np
import xarray as xr


def get_feed(
    samples: xr.DataArray,
    pattern: Sequence[str],
    /,
    *,
    offset: int = 0,
    origin: np.datetime64 | None = None,
) -> xr.DataArray:
    """Get the feed names of the VDIF samples.

    Args:
        samples: VDIF samples.
        pattern: Feed name pattern.
        offset: Offset of the feed name pattern.
        origin: Origin time for the feed name pattern.

    Returns:
        Feed names of the VDIF samples.

    """
    if origin is None:
        origin = samples.time[0]
    else:
        origin = np.datetime64(origin)

    dt = np.timedelta64(samples.ip_length, "ms")
    index = ((samples.time - origin) / dt).astype(int)

    return xr.DataArray(
        np.roll(pattern, offset)[index % len(pattern)],
        dims="time",
        coords={"time": samples.time},
        name="feed",
    )

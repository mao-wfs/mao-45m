__all__ = ["get_aggregated", "get_epl", "get_feed", "get_freq", "get_phase"]


# standard library
from collections.abc import Sequence


# dependencies
import numpy as np
import xarray as xr
from ndtools import Range
from numpy.typing import NDArray
from decode.stats import mean


# constants
C = 299_792_458  # m s^-1
FREQ_AT_CHAN0 = 16384e6  # Hz
FREQ_AT_CHAN8192 = 24576e6  # Hz
FREQ_STEP = 1e6  # Hz


def get_aggregated(
    samples: xr.DataArray,
    /,
    *,
    feed_origin: np.datetime64 | None = None,
    feed_pattern: NDArray[np.str_] | Sequence[str] = ("",),
    freq_binning: int = 8,
    freq_range: Range = Range(19.5e9, 22.5e9),
) -> xr.DataArray:
    """Get aggregated data (feed x freq) from VDIF samples.

    Args:
        samples: VDIF samples (time x chan).
        feed_origin: Origin time for the feed name pattern.
        feed_pattern: Feed name pattern to be repeated.
        freq_binning: Number of frequency channels to bin.
        freq_range: Frequency range (in Hz) to select.

    Returns:
        Aggregated data (feed x freq).

    """
    aggregated = (
        samples.groupby(get_feed(samples, feed_pattern, feed_origin))
        .mean("time")
        .rename("aggregated")
        .assign_coords(freq=get_freq(samples), time=samples.time[-1])
        .swap_dims(chan="freq")
    )
    aggregated = mean(aggregated, dim={"freq": freq_binning})
    return aggregated.sel(freq=aggregated.freq == freq_range)


def get_epl(aggregated: xr.DataArray, /) -> xr.DataArray:
    """Get the EPL (feed; in m) from the aggregated data."""
    return (
        np.arctan2(aggregated.imag, aggregated.real)
        .curvefit("freq", get_phase)
        .get("curvefit_coefficients")
        .sel(param="epl")
        .drop_vars("param")
        .rename("epl")
        .assign_attrs(units="m")
    )


def get_feed(
    samples: xr.DataArray,
    pattern: NDArray[np.str_] | Sequence[str] = ("",),
    origin: np.datetime64 | None = None,
    /,
) -> xr.DataArray:
    """Get the feed names from the VDIF samples.

    Args:
        samples: VDIF samples (time x chan).
        pattern: Feed name pattern to be repeated.
        origin: Origin time for the feed name pattern.

    Returns:
        Feed names of the VDIF samples (feed).

    """
    if origin is None:
        origin = samples.time[0]
    else:
        origin = np.datetime64(origin)

    dt = np.timedelta64(samples.ip_length, "ms")
    index = ((samples.time - origin) / dt).astype(int)

    return xr.DataArray(
        np.array(pattern)[index % len(pattern)],
        dims="time",
        coords={"time": samples.time},
        name="feed",
    )


def get_freq(samples: xr.DataArray, /) -> xr.DataArray:
    """Get the frequency (in Hz) of the VDIF samples."""
    freq = FREQ_AT_CHAN0 + FREQ_STEP * samples.chan
    freq[freq >= FREQ_AT_CHAN8192] = np.nan
    return freq.rename("freq").assign_attrs(units="Hz")


def get_phase(freq: NDArray, epl: float, /) -> NDArray:
    """Get the phase (in rad) from frequency (in Hz) and EPL (in m)"""
    return 2 * np.pi * epl * freq / C

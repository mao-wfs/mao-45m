__all__ = [
    "Word",
    "get_channel_number",
    "get_corr_data",
    "get_elapsed_seconds",
    "get_frame_number",
    "get_integ_time",
    "get_ip_length",
    "get_reference_epoch",
    "get_samples",
    "get_sample_number",
    "get_time",
    "get_thread_id",
    "get_word",
]


# standard library
from dataclasses import dataclass


# dependencies
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from typing_extensions import Self
from . import CHANS_PER_FRAME, CORR_DATA_BYTES, FRAMES_PER_SAMPLE, VDIF_HEAD_BYTES


@dataclass(frozen=True)
class Word:
    """Word parser for VDIF Head or Corr Head."""

    data: int

    def __getitem__(self, index: slice, /) -> int:
        """Get the slice of the word."""
        start, stop = index.start, index.stop
        return (self.data >> start) & ((1 << stop - start) - 1)

    @classmethod
    def from_bytes(cls, data: bytes, /) -> Self:
        """Create a Word parser from bytes."""
        return cls(int.from_bytes(data, "little"))


def get_channel_number(frame: bytes, /) -> int:
    """Get the channel number (1: CH0-255, ..., 64: CH16128-16383)."""
    return get_word(frame[:VDIF_HEAD_BYTES], 4)[16:24]


def get_corr_data(frame: bytes, /) -> NDArray[np.complex128]:
    """Get the complex auto or cross-correlation data."""
    data = np.frombuffer(frame[-CORR_DATA_BYTES:], dtype=np.int16)
    return data[0::2] + data[1::2] * 1j


def get_elapsed_seconds(frame: bytes, /) -> int:
    """Get the elapsed seconds from the reference epoch."""
    return get_word(frame[:VDIF_HEAD_BYTES], 0)[0:30]


def get_frame_number(frame: bytes, /) -> int:
    """Get the frame number within a second (0, 1, ...)."""
    return get_word(frame[:VDIF_HEAD_BYTES], 1)[0:24]


def get_integ_time(frame: bytes, /) -> float:
    """Get the integration time in seconds (0.005 or 0.010)."""
    return get_ip_length(frame) / 1000


def get_ip_length(frame: bytes, /) -> int:
    """Get the IP length (i.e. integration time) in ms (5 or 10)."""
    return get_word(frame[:VDIF_HEAD_BYTES], 4)[0:8]


def get_reference_epoch(frame: bytes, /) -> int:
    """Get the reference epoch (0: 2000 Jun, ..., 63: 2031 Jul.)."""
    return get_word(frame[:VDIF_HEAD_BYTES], 1)[24:30]


def get_samples(frames: list[bytes], /) -> xr.DataArray:
    """Get VDIF samples (time x chan) from VDIF frames."""
    data = []
    ids = []
    times = []

    for frame in frames:
        data.append(get_corr_data(frame))
        ids.append(get_channel_number(frame))
        times.append(get_time(frame))

    da = (
        xr.DataArray(
            data,
            dims=("time", "chan"),
            coords={
                "chan": np.arange(CHANS_PER_FRAME),
                "id": ("time", ids),
                "time": times,
            },
            attrs={
                "ip_length": get_ip_length(frames[0]),
            },
        )
        .set_index(temp=("id", "time"))
        .unstack("temp")
        .stack(temp=("chan", "id"), create_index=False)
    )

    return (
        da.assign_coords(temp=CHANS_PER_FRAME * (da.id - 1) + da.chan)
        .drop_vars(("chan", "id"))
        .rename(temp="chan")
        .sortby("time")
        .sortby("chan")
        .dropna("time")
    )


def get_sample_number(frame: bytes, /) -> int:
    """Get the sample number within a second (0, 1, ...)."""
    return get_frame_number(frame) // FRAMES_PER_SAMPLE


def get_time(frame: bytes, /) -> np.datetime64:
    """Get the recorded time of the frame in UTC."""
    return (
        np.datetime64("2000")
        + np.timedelta64(6 * get_reference_epoch(frame), "M")
        + np.timedelta64(get_elapsed_seconds(frame), "s")
        + np.timedelta64(get_sample_number(frame) * get_ip_length(frame), "ms")
        + np.timedelta64(0, "ns")
    )


def get_thread_id(frame: bytes, /) -> int:
    """Get the thread ID (1: IF 1x1, 2: IF 2x2, 5: IF 1x2, ...)."""
    return get_word(frame[:VDIF_HEAD_BYTES], 3)[16:26]


def get_word(head: bytes, n: int, /) -> Word:
    """Get the n-th word parser of a VDIF Head or Corr Head."""
    return Word.from_bytes(head[4 * n : 4 * (n + 1)])

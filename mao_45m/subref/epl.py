__all__ = ["calc_epl", "get_spectra", "setup_receiver"]


# standard library
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from logging import getLogger
from scipy.optimize import curve_fit
from struct import Struct
from threading import Event, Lock, Thread
from typing import Callable, NamedTuple


# dependencies
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from ..vdif.receive import get_socket


# constants
C = 299_792_458  # m/s
LITTLE_ENDIAN: str = "<"
LOCK = Lock()
LOGGER = getLogger(__name__)
LOWER_FREQ_MHZ = 16384
N_BYTES_PER_SCAN: int = 1312 * 64
N_BYTES_PER_UNIT: int = 1312
N_CHANS_FOR_FORMAT = 2048
N_ROWS_CORR_DATA: int = 512
N_ROWS_CORR_HEAD: int = 64
N_ROWS_VDIF_HEAD: int = 8
N_UNITS_PER_SCAN: int = 64
PACKET_BUFFER = deque(maxlen=6000)
REF_EPOCH_ORIGIN = np.datetime64("2000", "Y")
REF_EPOCH_UNIT = np.timedelta64(6, "M")
SHORT: str = "h"
TIME_PER_SCAN: float = 1e-2
UINT: str = "I"


# global variables
FEED = ["c", "t", "r", "b", "l"]
UDP_READY_EVENT = Event()


@contextmanager
def setup_receiver(
    *,
    group: str = "239.0.0.1",
    port: int = 11111,
) -> Iterator[None]:
    """Setup the UDP receiver for EPL calculation.

    Args:
        group: Multicast group address.
        port: Multicast port number.

    """

    sock = get_socket(group=group, port=port)

    receiver_thread = Thread(
        target=udp_receiver,
        args=(sock, UDP_READY_EVENT),
        daemon=True,
    )

    try:
        LOGGER.debug("Starting receiver thread...")
        receiver_thread.start()
        yield
    finally:
        LOGGER.debug("Finishing receiver thread...")
        receiver_thread.join(timeout=1.0)
        sock.close()


def get_spectra(
    feed_origin: str,
    feed_pattern: Sequence[str],
    freq_range: tuple[float, float],
    freq_binning: int = 8,
    size: int = 25,
) -> xr.DataArray:
    """Get Spectra.

    Args:
        feed_pattern: Feed name pattern to be repeated.
        freq_range: Frequency range to select (in Hz).
        freq_binning: Number of frequency channels to bin.
        size: Number of scan.

    """

    feed_origin = datetime.strptime(feed_origin, "%Y%m%dT%H%M%S")  # type: ignore
    COUNT = np.zeros(5, dtype=int)
    ch = int((freq_range[1] - freq_range[0]) * 1e-6 / freq_binning)  # MHz
    SPECTRA = np.zeros((5, ch), dtype=np.complex128)
    FEED_PATTERN = feed_pattern
    FREQ = get_freq(freq_binning)  # Hz
    FREQ_SELECTED = FREQ[(FREQ >= freq_range[0]) & (FREQ <= freq_range[1])]

    UDP_READY_EVENT.clear()
    UDP_READY_EVENT.wait()
    UDP_READY_EVENT.clear()
    scan = get_latest_packets(size)

    for i in range(size):
        frames = scan[i]
        data_time, spectrum = get_nth_spectrum_in_range(
            frames, FREQ, freq_range, freq_binning
        )
        if i == 0:
            start_time = data_time

        n = get_n_from_current_time(feed_origin, data_time)  # type: ignore
        target = FEED_PATTERN[n % len(FEED_PATTERN)]
        if target == "x":
            continue

        f = FEED.index(target)
        SPECTRA[f] += spectrum
        COUNT[f] += 1

    last_time = data_time

    LOGGER.debug(
        f"start_time={start_time}, last_time={last_time}, d_time={last_time-start_time}"
    )

    return xr.DataArray(
        data=[
            np.array(SPECTRA[0] / COUNT[0]),
            np.array(SPECTRA[1] / COUNT[1]),
            np.array(SPECTRA[2] / COUNT[2]),
            np.array(SPECTRA[3] / COUNT[3]),
            np.array(SPECTRA[4] / COUNT[4]),
        ],
        dims=("feed", "freq"),
        coords={
            "feed": FEED,
            "freq": FREQ_SELECTED,
            "time": last_time,
        },
    )


def calc_epl(spec: xr.DataArray) -> xr.DataArray:
    freq = spec.coords["freq"].values

    epl_dict = {}
    for f in ["c", "t", "r", "b", "l"]:
        epl_dict[f] = get_epl(spec.sel(feed=f).data, freq)

    return xr.DataArray(
        data=[
            epl_dict["c"],
            epl_dict["t"],
            epl_dict["r"],
            epl_dict["b"],
            epl_dict["l"],
        ],
        dims="feed",
        coords={"feed": FEED, "time": spec.coords["time"]},
    )


def udp_receiver(sock, udp_ready_event):
    while True:
        temp_buffer = []
        while True:
            frame, _ = sock.recvfrom(N_BYTES_PER_UNIT)
            array = np.frombuffer(
                frame,
                dtype=[
                    ("word_0", "u4"),
                    ("word_1", "u4"),
                    ("word_2", "u4"),
                    ("word_3", "u4"),
                    ("word_4", "u4"),
                    ("word_5", "u4"),
                    ("word_6", "u4"),
                    ("word_7", "u4"),
                    ("data", ("u1", 1280)),
                ],
            )
            word_4 = Word(array["word_4"])
            ch = word_4[16:24]
            if ch == 64:
                break

        while True:
            frame, _ = sock.recvfrom(N_BYTES_PER_UNIT)

            if len(frame) != N_BYTES_PER_UNIT:
                LOGGER.warning(
                    f" Received frame size anomaly: {len(frame)} bytes. Sample skipped."
                )
                break
            temp_buffer.append(frame)

            if len(temp_buffer) == N_UNITS_PER_SCAN:
                if not check_channel_order(temp_buffer):
                    LOGGER.warning(
                        "Channel sequence mismatch detected. Reinitializing reception."
                    )
                    break
                with LOCK:
                    PACKET_BUFFER.append(list(temp_buffer))
                temp_buffer.clear()
                udp_ready_event.set()


class Word:
    """VDIF header word parser."""

    def __init__(self, data: NDArray[np.int_]):
        self.data = data

    """VDIF header word as a 1D integer array."""

    def __getitem__(self, index: slice, /) -> NDArray[np.int_]:
        """Slice the VDIF header word."""
        start, stop = index.start, index.stop
        return (self.data >> start) & ((1 << stop - start) - 1)


class head_data(NamedTuple):
    time: datetime
    thread_id: NDArray[np.int_]
    ch: NDArray[np.int_]
    integ: NDArray[np.int_]


def read_head(frame: bytes) -> head_data:
    array = np.frombuffer(
        frame,
        dtype=[
            ("word_0", "u4"),
            ("word_1", "u4"),
            ("word_2", "u4"),
            ("word_3", "u4"),
            ("word_4", "u4"),
            ("word_5", "u4"),
            ("word_6", "u4"),
            ("word_7", "u4"),
            ("data", ("u1", 1280)),
        ],
    )
    word_0 = Word(array["word_0"])
    word_1 = Word(array["word_1"])
    word_3 = Word(array["word_3"])
    word_4 = Word(array["word_4"])
    seconds = int(word_0[0:30])
    frame_num = int(word_1[0:24])
    ref_epoch = int(word_1[24:30])
    thread_id = int(word_3[16:26])
    ch = int(word_4[16:24])
    integ = int(word_4[0:8])

    time = (
        REF_EPOCH_ORIGIN
        + REF_EPOCH_UNIT * ref_epoch
        + np.timedelta64(1, "s") * seconds
        + np.timedelta64(integ * (frame_num // 64), "ms")
    )
    time_dt = time.astype("datetime64[us]").astype(datetime)

    return head_data(time_dt, thread_id, ch, integ)  # type: ignore


def get_latest_packets(a: int) -> list:
    with LOCK:
        return list(PACKET_BUFFER)[-a:]


def check_channel_order(packet_set: list[bytes]) -> bool:
    ch_list = []

    for frame in packet_set:
        metadata = read_head(frame)
        ch_list.append(int(metadata.ch))
    expected = list(range(1, 65))
    if ch_list == expected:
        return True
    else:
        return False


# main features
def get_spectrum(
    scan: xr.Dataset,
    freq_binning: int = 8,
) -> tuple[datetime, np.ndarray]:
    n_integ = 1
    n_units = N_UNITS_PER_SCAN * n_integ
    n_chans = N_ROWS_CORR_DATA // 2

    spectra = np.empty([n_units, n_chans], dtype=complex)

    for i in range(n_units):
        frame = scan[i]
        time = read_head(frame).time  # type: ignore
        corr_data = read_corr_data(frame[288:1312])
        spectra[i] = parse_corr_data(corr_data)

    spectra = spectra.reshape([n_integ, N_UNITS_PER_SCAN * n_chans])
    spectrum = integrate_spectra(spectra, freq_binning)
    return time, spectrum


def get_nth_spectrum_in_range(
    scan: xr.Dataset,
    freq: np.ndarray,
    freq_range: tuple[float, float],
    freq_binning: int = 8,
) -> tuple[datetime, np.ndarray]:
    time, spec = get_spectrum(scan, freq_binning)
    filtered_spec = spec[(freq >= freq_range[0]) & (freq <= freq_range[1])]
    return time, filtered_spec


def get_epl(spec: np.ndarray, freq: np.ndarray) -> float:
    fit = curve_fit(line_through_origin, freq, get_phase(spec))
    slope = fit[0]
    slope = slope[0]
    epl = (C * slope) / (2 * np.pi)
    return epl


def get_freq(bin_width: int = 8, n_chans: int = 2048) -> np.ndarray:
    freq = 1e6 * (LOWER_FREQ_MHZ + np.arange(n_chans * bin_width))
    freq = freq.reshape((freq.shape[0] // bin_width, bin_width)).mean(-1)
    return freq


def get_n_from_current_time(start_time: datetime, data_time: datetime) -> int:
    dt = (data_time - start_time).total_seconds()
    n = int(round(dt / TIME_PER_SCAN)) - 1
    return n


def get_amp(da: np.ndarray) -> np.ndarray:
    """複素数DataArrayの絶対値Amplitudeを返す関数"""
    amp = np.abs(da)
    return amp


def get_phase(da: np.ndarray) -> np.ndarray:
    """複素数DataArrayの偏角(ラジアン単位)を返す関数"""
    phase = np.arctan2(da.imag, da.real)
    return phase


def line_through_origin(freq: np.ndarray, slope: float) -> np.ndarray:
    """原点を通る直線モデル"""
    return slope * freq


def integrate_spectra(spectra: np.ndarray, freq_binning: int = 8) -> np.ndarray:
    spectrum = spectra.mean(0)
    return spectrum.reshape([len(spectrum) // freq_binning, freq_binning]).mean(1)


# struct readers
def make_binary_reader(n_rows: int, dtype: str) -> Callable:
    struct = Struct(LITTLE_ENDIAN + dtype * n_rows)

    def reader(f):
        if isinstance(f, bytes):
            return struct.unpack(f)
        else:
            return struct.unpack(f.read(struct.size))

    return reader


read_vdif_head: Callable = make_binary_reader(N_ROWS_VDIF_HEAD, UINT)
read_corr_head: Callable = make_binary_reader(N_ROWS_CORR_HEAD, UINT)
read_corr_data: Callable = make_binary_reader(N_ROWS_CORR_DATA, SHORT)


# struct parsers
def parse_vdif_head(vdif_head: list):
    # not implemented yet
    pass


def parse_corr_head(corr_head: list):
    # not implemented yet
    pass


# 相関データ
def parse_corr_data(corr_data: list) -> np.ndarray:
    real = np.array(corr_data[0::2])
    imag = np.array(corr_data[1::2])
    return real + imag * 1j

__all__ = [
    "udp_receiver",
    "get_latest_packets",
    "generate_patterned",
    "get_nth_spectrum_in_range",
    "get_epl",
    "get_n_from_current_time",
    "get_freq",
]

# standard library
import re
from datetime import datetime, timedelta
from struct import Struct
from typing import Callable, Pattern, NamedTuple
from scipy.optimize import curve_fit
from collections import deque

# dependent packages
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from ..vdif.receive import get_socket


# constants
LITTLE_ENDIAN: str = "<"
UINT: str = "I"
SHORT: str = "h"
N_ROWS_VDIF_HEAD: int = 8
N_ROWS_CORR_HEAD: int = 64
N_ROWS_CORR_DATA: int = 512
N_UNITS_PER_SCAN: int = 64
N_BYTES_PER_UNIT: int = 1312  #
N_BYTES_PER_SCAN: int = 1312 * 64
TIME_PER_SCAN: float = 1e-2
TIME_FORMAT: str = "%Y%j%H%M%S%f"
VDIF_PATTERN: Pattern = re.compile(r"\w+_(\d+)_\d.vdif")
LOWER_FREQ_MHZ = 16384
N_CHANS_FOR_FORMAT = 2048
C = 299792458  # m/s
PI = np.pi
REF_EPOCH_ORIGIN = np.datetime64("2000", "Y")  # UTC
REF_EPOCH_UNIT = np.timedelta64(6, "M")


from threading import Event, Lock, Thread
from logging import getLogger

LOGGER = getLogger(__name__)
packet_buffer = deque(maxlen=6000)
lock = Lock()


feed = ["c", "t", "r", "b", "l"]
make_pattern = "xxxxx"
pattern_len = 0
freq = np.array([])
freq_selected = np.array([])
count = np.zeros(5, dtype=int)
spectra = np.zeros((5, 375), dtype=np.complex128)
udp_ready_event = Event()


def setup(pattern, chbin, dest_addr, dest_port, group) -> None:
    global make_pattern, pattern_len, freq, freq_selected

    sock = get_socket(port=dest_port, group=group)
    n_offset_2024 = 2  #  #2024年のオフセット
    make_pattern = generate_patterned(pattern, n_offset_2024)
    pattern_len = len(make_pattern)
    freq = get_freq(chbin)
    freq_selected = freq[(freq >= 19.5) & (freq <= 22.5)]

    receiver_thread = Thread(
        target=udp_receiver, args=(sock, udp_ready_event), daemon=True
    )
    receiver_thread.start()
    LOGGER.debug("Starting receiver thread...")


def get_spectra(d0: str, chbin: int, delay: float, a: int) -> xr.Dataset:
    global udp_ready_event

    d0 = datetime.strptime(d0, "%Y%m%dT%H%M%S")  # type: ignore
    udp_ready_event.clear()
    udp_ready_event.wait()
    udp_ready_event.clear()
    scan = get_latest_packets(a)

    for i in range(a):
        frames = scan[i]
        data_time, spectrum = get_nth_spectrum_in_range(frames, freq, chbin)
        if i == 0:
            start_time = data_time

        n = get_n_from_current_time(d0, data_time, delay)  # type: ignore
        target = make_pattern[n % pattern_len]
        if target == "x":
            continue

        f = feed.index(target)
        spectra[f] += spectrum
        count[f] += 1

    last_time = data_time
    middle_time = timedelta(seconds=a * TIME_PER_SCAN / 2)
    time_1 = last_time - middle_time
    time_2 = start_time + middle_time
    time_3 = (last_time - start_time) / 2 + start_time

    LOGGER.debug(f"time_1={time_1}, time_2={time_2}, time_3={time_3}")

    ds = xr.Dataset(
        {
            "c": (("time", "freq"), np.array(spectra[0] / count[0])[np.newaxis, :]),
            "t": (("time", "freq"), np.array(spectra[1] / count[1])[np.newaxis, :]),
            "r": (("time", "freq"), np.array(spectra[2] / count[2])[np.newaxis, :]),
            "b": (("time", "freq"), np.array(spectra[3] / count[3])[np.newaxis, :]),
            "l": (("time", "freq"), np.array(spectra[4] / count[4])[np.newaxis, :]),
        },
        coords={
            "time": np.array([time_1], dtype="M8"),
            "freq": np.array(freq_selected),
        },
    )

    return ds


def calc_epl(spec: xr.Dataset) -> xr.Dataset:
    freq = spec.coords["freq"].values

    epl_dict = {}
    for f in ["c", "t", "r", "b", "l"]:
        epl_dict[f] = get_epl(np.ravel(spec[f].values), freq)

    ds = xr.Dataset(
        {
            "c": (("time",), np.array([epl_dict["c"]])),
            "t": (("time",), np.array([epl_dict["t"]])),
            "r": (("time",), np.array([epl_dict["r"]])),
            "b": (("time",), np.array([epl_dict["b"]])),
            "l": (("time",), np.array([epl_dict["l"]])),
        }
    )

    return ds


def udp_receiver(sock, udp_ready_event):

    LOGGER.debug("デーモン開始")
    while True:
        temp_buffer = []
        # 最初の受信処理
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
                LOGGER.debug(f"受信フレームサイズ異常: {len(frame)} bytes (スキップ)")
                break
            temp_buffer.append(frame)

            if len(temp_buffer) == N_UNITS_PER_SCAN:
                if not check_channel_order(temp_buffer):
                    LOGGER.debug("⚠️ チャンネル順異常のため、最初から受信し直します")
                    break
                with lock:
                    packet_buffer.append(list(temp_buffer))
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
    with lock:
        return list(packet_buffer)[-a:]


def check_channel_order(packet_set: list[bytes]) -> bool:
    ch_list = []

    for frame in packet_set:
        metadata = read_head(frame)
        ch_list.append(int(metadata.ch))
    expected = list(range(1, 65))
    if ch_list == expected:
        return True
    else:
        LOGGER.debug("⚠️ チャンネル順に異常あり！")
        return False


# main features
def get_spectrum(
    scan: xr.Dataset,
    chbin: int = 8,
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
    spectrum = integrate_spectra(spectra, chbin)
    return time, spectrum


def get_nth_spectrum_in_range(
    scan: xr.Dataset, freq: np.ndarray, chbin: int = 8
) -> tuple[datetime, np.ndarray]:
    time, spec = get_spectrum(scan, chbin)
    filtered_spec = spec[(freq >= 19.5) & (freq <= 22.5)]
    return time, filtered_spec


def get_epl(spec: np.ndarray, freq: np.ndarray) -> float:
    fit = curve_fit(line_through_origin, freq, get_phase(spec))
    slope = fit[0]
    slope = slope[0]
    epl = (C * slope * 1e-9) / (2 * PI)
    return epl


def get_freq(bin_width: int = 8, n_chans: int = 2048) -> np.ndarray:
    freq = 1e-3 * (LOWER_FREQ_MHZ + np.arange(n_chans * bin_width))
    freq = freq.reshape((freq.shape[0] // bin_width, bin_width)).mean(-1)
    return freq


# パターン生成
def generate_patterned(pattern: str, offset: int = 0) -> str:
    # 文字列としてロール
    result = pattern[offset:] + pattern[:offset]
    return result


def get_n_from_current_time(
    start_time: datetime, data_time: datetime, delay: float = 0.0
) -> int:
    dt = (data_time - start_time - timedelta(seconds=delay)).total_seconds()
    n = int(round(dt / TIME_PER_SCAN)) - 1
    return n


# 振幅
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


def integrate_spectra(spectra: np.ndarray, chbin: int = 8) -> np.ndarray:
    spectrum = spectra.mean(0)
    return spectrum.reshape([len(spectrum) // chbin, chbin]).mean(1)


# struct readers
def make_binary_reader(n_rows: int, dtype: str) -> Callable:
    struct = Struct(LITTLE_ENDIAN + dtype * n_rows)

    def reader(f):
        # fがbytesならそのまま、ファイルオブジェクトならread
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
    # not implemented yet ここにvdifヘッダの解析処理を実装する
    pass


def parse_corr_head(corr_head: list):
    # not implemented yet
    pass


# 相関データ
def parse_corr_data(corr_data: list) -> np.ndarray:
    real = np.array(corr_data[0::2])  # 偶数の要素を実部
    imag = np.array(corr_data[1::2])  # 奇数の要素を虚部
    return real + imag * 1j

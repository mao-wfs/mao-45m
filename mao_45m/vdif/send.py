__all__ = ["send"]


# standard library
from logging import getLogger
from os import PathLike
from pathlib import Path
from socket import AF_INET, IPPROTO_IP, IP_MULTICAST_TTL, SOCK_DGRAM, socket
from time import perf_counter


# dependencies
from tqdm import trange
from . import FRAMES_PER_SAMPLE, FRAME_BYTES
from .convert import get_ip_length


# constants
LOGGER = getLogger(__name__)


def send(
    vdif: PathLike[str] | str,
    /,
    *,
    group: str = "239.0.0.1",
    port: int = 11111,
    repeat: bool = False,
    status: bool | int = False,
    ttl: int = 1,
) -> None:
    """Send a VDIF file over UDP multicast.

    Args:
        vdif: Path to the VDIF file.
        group: Multicast group address.
        port: Multicast port number.
        repeat: Whether to repeat sending the VDIF file.
        status: Whether to show the sending status.
        ttl: Time-to-live for multicast packets.

    """
    n_frames, remainder = divmod(Path(vdif).stat().st_size, FRAME_BYTES)

    if remainder:
        LOGGER.warning(f"VDIF file is truncated ({remainder} bytes remaining).")

    with socket(AF_INET, SOCK_DGRAM) as sock:
        sock.setsockopt(IPPROTO_IP, IP_MULTICAST_TTL, ttl)

        def send_once() -> None:
            with open(vdif, "rb") as file:
                for _ in trange(
                    n_frames,
                    desc=f"Sending {vdif}",
                    disable=not status,
                    position=int(status) - 1,
                    unit="frame",
                    unit_scale=True,
                ):
                    start = perf_counter()
                    sock.sendto(frame := file.read(FRAME_BYTES), (group, port))
                    seconds_per_frame = get_ip_length(frame) / 1000 / FRAMES_PER_SAMPLE
                    end = perf_counter()
                    sleep(seconds_per_frame - (end - start))

        try:
            if repeat:
                while True:
                    send_once()
            else:
                send_once()
        except KeyboardInterrupt:
            LOGGER.warning("Sending interrupted by user.")


def sleep(seconds: float, /) -> None:
    """Busy-waiting sleep (more accurate but more CPU usage)."""
    start = perf_counter()
    end = start + seconds

    while perf_counter() < end:
        pass

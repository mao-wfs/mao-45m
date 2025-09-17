__all__ = ["receive"]


# standard library
from logging import getLogger
from os import PathLike
from socket import (
    AF_INET,
    INADDR_ANY,
    IPPROTO_IP,
    IP_ADD_MEMBERSHIP,
    SOCK_DGRAM,
    SOL_SOCKET,
    SO_REUSEADDR,
    inet_aton,
    socket,
)
from struct import pack


# dependencies
from tqdm import tqdm
from . import VDIF_FRAME_BYTES


# constants
LOGGER = getLogger(__name__)


def receive(
    vdif: PathLike[str] | str,
    /,
    *,
    group: str = "239.0.0.1",
    port: int = 11111,
    progress: bool = False,
) -> None:
    """Receive a VDIF file over UDP multicast.

    Args:
        vdif: Path to the VDIF file.
        group: Multicast group address.
        port: Multicast port number.
        progress: Whether to show a progress bar.

    """
    with socket(AF_INET, SOCK_DGRAM) as sock:
        sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        sock.bind(("", port))

        mreq = pack("4sL", inet_aton(group), INADDR_ANY)
        sock.setsockopt(IPPROTO_IP, IP_ADD_MEMBERSHIP, mreq)

        try:
            with (
                open(vdif, "wb") as file,
                tqdm(
                    desc=f"Receiving {vdif}",
                    disable=not progress,
                    unit="B",
                    unit_scale=True,
                ) as bar,
            ):
                while True:
                    frame, _ = sock.recvfrom(VDIF_FRAME_BYTES)
                    bar.update(file.write(frame))
        except KeyboardInterrupt:
            LOGGER.warning("Receiving interrupted by user.")

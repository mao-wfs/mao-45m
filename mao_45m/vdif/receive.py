__all__ = ["Frames", "get_frames", "get_socket", "receive"]


# standard library
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
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
from threading import Event, Lock, Thread

try:
    from socket import SO_REUSEPORT  # type: ignore
except ImportError:
    SO_REUSEPORT = None  # type: ignore


# dependencies
from tqdm import tqdm
from . import FRAME_BYTES


# constants
LOGGER = getLogger(__name__)


class Frames:
    """Thread-safe deque for VDIF frames.

    Args:
        size: Maximum number of the latest frames to keep.

    """

    def __init__(self, size: int = 100000, /) -> None:
        self.data = deque(maxlen=size)
        self.lock = Lock()

    def append(self, frame: bytes, /) -> None:
        """Append a frame to the right end of the deque."""
        with self.lock:
            self.data.append(frame)

    def appendleft(self, frame: bytes, /) -> None:
        """Append a frame to the left end of the deque."""
        with self.lock:
            self.data.appendleft(frame)

    def get(self, n: int, /) -> list[bytes]:
        """Get the n frames from the right end of the deque."""
        with self.lock:
            n = min(n, len(self.data))
            return [self.data[i] for i in range(-n, 0)]

    def getleft(self, n: int, /) -> list[bytes]:
        """Get the n frames from the left end of the deque."""
        with self.lock:
            n = min(n, len(self.data))
            return [self.data[i] for i in range(0, n)]

    def pop(self, n: int, /) -> list[bytes]:
        """Pop the n frames from the right end of the deque."""
        with self.lock:
            n = min(n, len(self.data))
            return [self.data.pop() for _ in range(n)][::-1]

    def popleft(self, n: int, /) -> list[bytes]:
        """Pop the n frames from the left end of the deque."""
        with self.lock:
            n = min(n, len(self.data))
            return [self.data.popleft() for _ in range(n)]

    def __len__(self) -> int:
        """Get the number of frames in the deque."""
        with self.lock:
            return len(self.data)


def receive(
    vdif: PathLike[str] | str,
    /,
    *,
    group: str = "239.0.0.1",
    port: int = 11111,
    size: int = 100000,
    status: bool | int = False,
    usage: bool | int = False,
) -> None:
    """Receive a VDIF file over UDP multicast.

    Args:
        vdif: Path to the VDIF file.
        group: Multicast group address.
        port: Multicast port number.
        size: Maximum number of the latest frames to keep.
        status: Whether to show the receiving status.
        usage: Whether to show the buffer usage.

    """
    with get_frames(size, group=group, port=port, usage=usage) as frames:
        try:
            with (
                open(vdif, "wb") as file,
                tqdm(
                    desc=f"Receiving {vdif}",
                    disable=not status,
                    position=int(status) - 1,
                    unit="B",
                    unit_scale=True,
                ) as bar,
            ):
                while True:
                    try:
                        bar.update(file.write(frames.popleft(1)[0]))
                    except Exception:
                        pass
        except KeyboardInterrupt:
            LOGGER.warning("Receiving interrupted by user.")


@contextmanager
def get_frames(
    size: int = 100000,
    /,
    *,
    group: str = "239.0.0.1",
    port: int = 11111,
    usage: bool | int = False,
) -> Iterator[Frames]:
    """Get streamed VDIF frames over UDP multicast.

    Args:
        size: Maximum number of the latest frames to keep.
        group: Multicast group address.
        port: Multicast port number.
        usage: Whether to show the deque usage.

    Yields:
        Frames: Thread-safe deque for VDIF frames.

    """
    with get_socket(group=group, port=port) as sock:
        frames = Frames(size)
        event = Event()

        def receive():
            with tqdm(
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                desc=f"Deque usage",
                disable=not usage,
                position=int(usage) - 1,
                total=size,
                unit="%",
            ) as bar:
                while not event.is_set():
                    frame, _ = sock.recvfrom(FRAME_BYTES)
                    frames.append(frame)
                    bar.n = len(frames)
                    bar.refresh()

        thread = Thread(target=receive, daemon=True)

        try:
            thread.start()
            yield frames
        finally:
            event.set()
            thread.join(1)


def get_socket(*, group: str = "239.0.0.1", port: int = 11111) -> socket:
    """Get a socket object for UDP multicast."""
    sock = socket(AF_INET, SOCK_DGRAM)
    sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

    if SO_REUSEPORT is not None:
        sock.setsockopt(SOL_SOCKET, SO_REUSEPORT, 1)

    sock.bind(("", port))

    mreq = pack("4sL", inet_aton(group), INADDR_ANY)
    sock.setsockopt(IPPROTO_IP, IP_ADD_MEMBERSHIP, mreq)
    return sock

__all__ = [
    "VDIF_FRAME_BYTES",
    "VDIF_HEAD_BYTES",
    "CORR_HEAD_BYTES",
    "CORR_DATA_BYTES",
    "reader",
    "receiver",
    "sender",
]


# constants
VDIF_FRAME_BYTES = 1312
VDIF_HEAD_BYTES = 32
CORR_HEAD_BYTES = 256
CORR_DATA_BYTES = 1024
FRAMES_PER_SAMPLE = 64


# dependencies
from . import reader
from . import receiver
from . import sender

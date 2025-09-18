__all__ = [
    "FRAME_BYTES",
    "VDIF_HEAD_BYTES",
    "CORR_HEAD_BYTES",
    "CORR_DATA_BYTES",
    "CHANS_PER_FRAME",
    "FRAMES_PER_SAMPLE",
    "convert",
    "receive",
    "send",
]


# constants
FRAME_BYTES = 1312
VDIF_HEAD_BYTES = 32
CORR_HEAD_BYTES = 256
CORR_DATA_BYTES = 1024
CHANS_PER_FRAME = 256
FRAMES_PER_SAMPLE = 64


# dependencies
from . import convert
from . import receive
from . import send

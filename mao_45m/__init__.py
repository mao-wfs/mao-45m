__all__ = ["cosmos", "subref", "vdif", "utils"]
__version__ = "0.6.0"


# dependencies
from fire import Fire
from . import cosmos
from . import subref
from . import vdif
from . import utils


def main():
    Fire(
        {
            "cosmos": {
                "receive": cosmos.receive,
                "send": cosmos.send,
            },
            "subref": {
                "control": subref.control.control,
            },
            "vdif": {
                "receive": vdif.receive.receive,
                "send": vdif.send.send,
            },
        }
    )

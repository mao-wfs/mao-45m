__all__ = ["subref", "vdif", "utils"]


# dependencies
from fire import Fire
from . import subref
from . import vdif
from . import utils


def main():
    Fire(
        {
            "subref": {"run": subref.run},
            "vdif": {
                "receive": vdif.receive.receive,
                "send": vdif.send.send,
            },
        }
    )

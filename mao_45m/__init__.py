__all__ = ["subref", "vdif"]


# dependencies
from fire import Fire
from . import subref
from . import vdif


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

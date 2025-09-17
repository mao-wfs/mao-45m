__all__ = ["subref", "vdif"]


# dependencies
from fire import Fire
from . import subref
from . import vdif


def main():
    Fire(
        {
            "subref": {"run": subref.run},
            "vdif": {"send": vdif.sender.send},
        }
    )

__all__ = ["subref"]


# dependencies
from fire import Fire
from . import subref


def main():
    Fire({"subref": {"run": subref.run}})

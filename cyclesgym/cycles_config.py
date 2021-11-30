import pathlib


__all__ = ['CYCLES_DIR']


CYCLES_DIR = pathlib.Path(__file__).parent.parent.joinpath('cycles')
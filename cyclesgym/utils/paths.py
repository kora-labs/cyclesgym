import pathlib

__all__ = ['PROJECT_PATH', 'CYCLES_PATH', 'AGENTS_PATH', 'FIGURES_PATH', 'DATA_PATH']

PROJECT_PATH = pathlib.Path(__file__).parents[2]
CYCLES_PATH = PROJECT_PATH.joinpath('cycles')
AGENTS_PATH = PROJECT_PATH.joinpath('agents')
FIGURES_PATH = PROJECT_PATH.joinpath('figures')
DATA_PATH = PROJECT_PATH.joinpath('data')
TEST_PATH = PROJECT_PATH.joinpath('cyclesgym', 'tests')

CYCLES_PATH.mkdir(exist_ok=True, parents=True)
AGENTS_PATH.mkdir(exist_ok=True, parents=True)
FIGURES_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH.mkdir(exist_ok=True, parents=True)

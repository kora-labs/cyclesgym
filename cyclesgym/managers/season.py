import pandas as pd

from cyclesgym.managers.common import Manager
from cyclesgym.managers.utils import date_to_ydoy


__all__ = ['SeasonManager']


class SeasonManager(Manager):
    def __init__(self, fname=None):
        self.season_df = pd.DataFrame()
        super().__init__(fname)

    def _parse(self):
        if self.fname is not None:
            with open(self.fname, 'r') as f:
                for i, l in enumerate(f.readlines()):
                    if i == 0:
                        columns = [n.strip(' \n') for n in l.split('\t')]
                    elif i == 2:
                        value = [[float(v) if v.replace('.', '', 1).isdigit()
                                 else v for v in l.split()]]
            self.season_df = pd.DataFrame(data=value, index=None, columns=columns)
            date_to_ydoy(self.season_df, old_col_name='DATE',
                         new_col_names=['YEAR', 'DOY'], inplace=True)
            date_to_ydoy(self.season_df, old_col_name='PLANT_DATE',
                         new_col_names=['PLANT_YEAR', 'PLANT_DOY'],
                         inplace=True)
import numpy as np
import pandas as pd

from cyclesgym.managers.common import InputFileManager
from cyclesgym.managers.utils import num_lines


__all__ = ['WeatherManager']


class WeatherManager(InputFileManager):
    def __init__(self, fname=None):
        self.immutables = pd.DataFrame()
        self.mutables = pd.DataFrame
        super().__init__(fname)

    def _parse(self, fname):
        self._parse_immutable(fname)
        self._parse_mutables(fname)

    def _parse_immutable(self, fname):
        immutables = {}
        with open(fname, 'r') as f:
            for _ in range(3):
                line = next(f)
                immutables.update({line.split()[0]: [float(line.split()[1])]})
            self.immutables = pd.DataFrame(immutables)

    def _parse_mutables(self, fname):
        n = num_lines(fname)
        values = np.zeros((n-5, 9))

        with open(fname, 'r') as f:
            for i, l in enumerate(f.readlines()):
                if i < 3 or i == 4:
                    pass
                elif i == 3:
                    columns = l.split()
                else:
                    values[i - 5, :] = [float(j) if i >= 2 else int(j) for i, j in enumerate(l.split())]

        self.mutables = pd.DataFrame(data=values, index=None, columns=columns)
        self.mutables = self.mutables.astype({'YEAR': int,
                                              'DOY': int})

    def _to_str_immutables(self):
        s = ''
        for (name, data) in self.immutables.iteritems():
            s += f'{name:<20}{data.values[0]}\n'
        return s

    def _to_str_mutables(self):
        return self.mutables.to_csv(index=False, sep=' ')

    def _to_str(self):
        return f'{self._to_str_immutables()}{self._to_str_mutables()}'

    def get_day(self, year, doy):
        return self.mutables.loc[(self.mutables['YEAR'] == year) & (self.mutables['DOY'] == doy)]
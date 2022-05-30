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
        if fname is not None:
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
        values = np.zeros((n, 9))

        lines_to_skip = []
        with open(fname, 'r') as f:
            for i, l in enumerate(f.readlines()):
                if l.startswith(('#', 'LATITUDE', 'ALTITUDE',
                                 'SCREENING_HEIGHT')):
                    lines_to_skip.append(i)
                    pass
                elif l.startswith('YEAR'):
                    columns = l.split()
                    lines_to_skip.append(i)
                else:
                    values[i, :] = [float(j) if i >= 2 else int(j) for i, j in enumerate(l.split())]
            values = np.delete(values, lines_to_skip, 0)

        self.mutables = pd.DataFrame(data=values, index=None, columns=columns)
        self.mutables = self.mutables.astype({'YEAR': int,
                                              'DOY': int})

    def _to_str_immutables(self):
        s = ''
        for (name, data) in self.immutables.iteritems():
            s += f'{name:<20}{data.values[0]}\n'
        return s

    def _to_str_mutables(self):
        # units_of_measurement = '####    ###     mm      deg C   deg C   MJ/m2   %       %       m/s\n'
        return self.mutables.to_csv(index=False, sep=' ')

    def _to_str(self):
        return f'{self._to_str_immutables()}{self._to_str_mutables()}'

    def get_day(self, year, doy):
        return self.mutables.loc[(self.mutables['YEAR'] == year) & (self.mutables['DOY'] == doy)]

    @classmethod
    def from_df(cls, immutables_df, mutables_df):
        # TODO: Perform sanity checks on df
        manager = cls(fname=None)
        manager.immutables = immutables_df.copy()
        manager.mutables = mutables_df.copy()
        return manager
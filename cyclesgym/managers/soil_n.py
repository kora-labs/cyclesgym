import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

from cyclesgym.managers.common import Manager
from cyclesgym.managers.utils import date_to_ydoy, ydoy_to_date

__all__ = ['SoilNManager']


class SoilNManager(Manager):
    def __init__(self, fname=None):
        self.soil_n_state = pd.DataFrame()
        super().__init__(fname)

    def _valid_input_file(self, fname):
        if fname is None:
            return True
        else:
            return super(SoilNManager, self)._valid_input_file(fname) and fname.suffix == '.dat'

    def _valid_output_file(self, fname):
        return super(SoilNManager, self)._valid_output_file(fname) and fname.suffix == '.dat'

    def _parse(self, fname):
        if fname is not None:
            if os.path.isfile(fname) and os.path.getsize(fname) > 0:
                # Read and remove unit of measurement row
                df = pd.read_csv(fname, sep='\t').drop(index=0)
                df.reset_index(drop=True, inplace=True)

                # Remove empty spaces and cast as floats
                df.columns = df.columns.str.strip(' ')
                numeric_cols = df.columns[1:]
                df[numeric_cols] = df[numeric_cols].astype(float)

                # Convert date to be consistent with weather
                date_to_ydoy(df, old_col_name='DATE',
                             new_col_names=['YEAR', 'DOY'], inplace=True)
                self.soil_n_state = df

    def _to_str(self):
        # Not necessary since this is not an input file but I had already implemented it
        return ydoy_to_date(self.soil_n_state, old_col_names=['YEAR', 'DOY'],
                            new_col_name='DATE').to_csv(index=False, sep='\t')

    def __str__(self):
        return self._to_str()

    def get_day(self, year, doy):
        if 'YEAR' in self.soil_n_state.columns and 'DOY' in self.soil_n_state.columns:
            return self.soil_n_state.loc[(self.soil_n_state['YEAR'] == year) & (self.soil_n_state['DOY'] == doy)]
        else:
            return pd.DataFrame()

    def plot(self, columns):
        columns = np.atleast_2d(columns)
        shape = columns.shape
        fig, axes = plt.subplots(*shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                axes[i, j].plot(self.soil_n_state[columns[i, j]])
        plt.show()
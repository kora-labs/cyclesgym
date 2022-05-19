import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cyclesgym.managers.common import Manager
from cyclesgym.managers.utils import date_to_ydoy, ydoy_to_date

__all__ = ['CropManager']


class CropManager(Manager):
    def __init__(self, fname=None):
        self.crop_state = pd.DataFrame()
        super().__init__(fname)

    def _valid_input_file(self, fname):
        if fname is None:
            return True
        else:
            return super(CropManager, self)._valid_input_file(fname) and fname.suffix == '.dat'

    def _valid_output_file(self, fname):
        return super(CropManager, self)._valid_output_file(fname) and fname.suffix == '.dat'

    def _parse(self, fname):
        if fname is not None:
            # Read and remove unit of measurement row
            df = pd.read_csv(fname, sep='\t').drop(index=0)
            df.reset_index(drop=True, inplace=True)

            # Remove empty spaces and cast as floats
            df.columns = df.columns.str.strip(' ')
            df['CROP'] = df['CROP'].str.strip(' ')
            df['STAGE'] = df['STAGE'].str.strip(' ')
            numeric_cols = df.columns[3:]
            df[numeric_cols] = df[numeric_cols].astype(float)

            # Convert date to be consistent with weather
            date_to_ydoy(df, old_col_name='DATE',
                         new_col_names=['YEAR', 'DOY'], inplace=True)
            self.crop_state = df

    def _to_str(self):
        # Not necessary since this is not an input file but I had already implemented it
        return ydoy_to_date(self.crop_state, old_col_names=['YEAR', 'DOY'],
                            new_col_name='DATE').to_csv(index=False, sep='\t')
    def __str__(self):
        return self._to_str()

    def get_day(self, year, doy):
        return self.crop_state.loc[(self.crop_state['YEAR'] == year) & (self.crop_state['DOY'] == doy)]

    def plot(self, columns):
        columns = np.atleast_2d(columns)
        shape = columns.shape
        fig, axes = plt.subplots(*shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                axes[i, j].plot(self.crop_state[columns[i, j]])
        plt.show()
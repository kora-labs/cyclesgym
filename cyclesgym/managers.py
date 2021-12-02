from pathlib import Path
import numpy as np
import warnings
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

import pandas as pd

__all__ = ['OperationManager', 'ControlManager', 'WeatherManager', 'CropManager']

OPERATION_TYPES = ('TILLAGE', 'PLANTING', 'FIXED_FERTILIZATION', 'FIXED_IRRIGATION', 'AUTO_IRRIGATION')


class Manager(ABC):
    def __init__(self, fname=None):
        self.fname = fname
        if not self._valid_input_file():
            raise ValueError('File not existing. To initialize manager without a file, pass None as input')
        self._parse()

    def _valid_input_file(self):
        if self.fname is not None:
            return self.fname.is_file()
        else:
            return True

    @abstractmethod
    def _parse(self):
        pass

    def update(self, fname):
        self.fname = fname
        if not self._valid_input_file():
            raise ValueError(f'{self.fname}File not existing. To initialize manager without a file, pass None as input')
        self._parse()


class OperationManager(object):
    def __init__(self, fname=None):
        self.fname = fname
        self.op_dict = dict()
        self._parse()

    def _parse(self):
        if self.fname is not None:
            with open(self.fname, 'r') as f:
                line_n = None
                k = None
                self.op_dict = dict()
                # TODO: using (year, doy, operation_type) as key, creates conflicts when there is fertilization applied to different layers on the same day
                for line in f.readlines():
                    if line.startswith(OPERATION_TYPES):
                        operation = line.strip(' ')
                        line_n = 0
                        if k is not None:
                            self.op_dict.update({k: v})
                    if line_n is not None:
                        if line_n == 1:
                            year = int(line.split()[1])
                        if line_n == 2:
                            doy = int(line.split()[1])
                            k = (year, doy, operation)
                            v = dict()
                        if line_n > 2:
                            if len(line.split()) > 0:
                                argument = line.split()[0]
                                value = line.split()[1]
                                v.update({argument: value})
                        line_n += 1
                if k is not None:
                    self.op_dict.update({k: v})
        else:
            self.op_dict = {}

    def count_same_day_events(self, year, doy):
        return len(self.get_same_day_events(year, doy))

    def get_same_day_events(self, year, doy):
        return {k: v for k, v in self.op_dict.items() if k[0] == year and k[1] == doy}

    def sort_operation(self):
        self.op_dict = dict(sorted(self.op_dict.items()))

    def to_string(self):
        s = ''
        for k, v in self.op_dict.items():
            year, doy, operation = k
            s += operation + f"\n{'YEAR':30}\t{year}\n{'DOY':30}\t{doy}\n"
            for argument, value in v.items():
                if argument == 'operation':
                    pass
                else:
                    s += f'{argument:30}\t{value}\n'
            s += '\n'
        return s

    def __str__(self):
        return self.to_string()

    def _insert_single_operation(self, op_key, op_val, force=True):
        year, doy, operation = op_key
        assert operation in OPERATION_TYPES, f'Operation must be one of the following {OPERATION_TYPES}'
        collisions = [operation == k[2] for k in self.get_same_day_events(year, doy).keys()]
        if any(collisions):
            warnings.warn(f'There is already an operation {operation} for they day {doy} of year {year}.')
            if force:
                self.op_dict.update({op_key: op_val})
            else:
                pass
        else:
            self.op_dict.update({op_key: op_val})

    def insert_new_operations(self, op, force=True):
        for k, v in op.items():
            self._insert_single_operation(k, v, force=force)
        self.sort_operation()

    def _delete_single_operation(self, k):
        self.op_dict.pop(k)

    def delete_operations(self, keys):
        for k in keys:
            self._delete_single_operation(k)

    def save(self, fname=None):
        if fname is None:
            if self.fname is None:
                raise RuntimeError('A file name must be provided')
            else:
                # Overwrite the initial file
                fname = self.fname
        self.sort_operation()
        s = self.to_string()
        with open(fname, 'w') as fp:
            fp.write(s)


class CropManager(Manager):
    def __init__(self, fname=None):
        self.crop_state = pd.DataFrame()
        super().__init__(fname)

    def _parse(self):
        if self.fname is not None:
            # Read and remove unit of measurement row
            df = pd.read_csv(self.fname, sep='\t').drop(index=0)
            df.reset_index(drop=True, inplace=True)

            # Remove empty spaces and cast as floats
            df.columns = df.columns.str.strip(' ')
            df['CROP'] = df['CROP'].str.strip(' ')
            df['STAGE'] = df['STAGE'].str.strip(' ')
            numeric_cols = df.columns[3:]
            df[numeric_cols] = df[numeric_cols].astype(float)

            # Convert date to be consistent with weather
            df.insert(1, 'DOY', pd.to_datetime(df['DATE']).dt.dayofyear)
            df.insert(1, 'YEAR', pd.to_datetime(df['DATE']).dt.year)
            df.drop(columns='DATE', inplace=True)
            self.crop_state = df

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


class WeatherManager(Manager):
    def __init__(self, fname=None):
        self.immutables = pd.DataFrame()
        self.mutables = pd.DataFrame
        super().__init__(fname)

    def _parse(self):
        self._parse_immutable()
        self._parse_mutables()

    def _parse_immutable(self):
        immutables = {}
        with open(self.fname, 'r') as f:
            for _ in range(3):
                line = next(f)
                immutables.update({line.split()[0]: [float(line.split()[1])]})
            self.immutables = pd.DataFrame(immutables)

    def _parse_mutables(self):
        n = num_lines(self.fname)
        values = np.zeros((n-5, 9))

        with open(self.fname, 'r') as f:
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

    def get_day(self, year, doy):
        return self.mutables.loc[(self.mutables['YEAR'] == year) & (self.mutables['DOY'] == doy)]


class ControlManager(Manager):
    def __init__(self, fname=None):
        self.fname = fname
        self.ctrl_dict = dict()
        self._non_numeric = ['CROP_FILE', 'OPERATION_FILE', 'SOIL_FILE', 'WEATHER_FILE', 'REINIT_FILE']
        super().__init__(fname)

    def _parse(self):
        if self.fname is not None:
            with open(self.fname, 'r') as f:
                for line in f.readlines():
                    if line.startswith(('\n', '#')):
                        pass
                    else:
                        k, v = line.split()[0:2]
                        v = v if k in self._non_numeric else int(v)
                        self.ctrl_dict.update({k: v})

    def to_string(self):
        s = '## SIMULATION YEARS ##\n\n'
        for k, v in self.ctrl_dict.items():
            s += f"{k:30}\t{v}\n"
            if k.startswith('ROTATION_SIZE'):
                s += '\n## SIMULATION OPTIONS ##\n\n'
            elif k.startswith('ANNUAL_NFLUX_OUT'):
                s += '\n## OTHER INPUT FILES ##\n\n'
        return s

    def __str__(self):
        return self.to_string()


def num_lines(file):
    with open(file, 'r') as f:
        for i, l in enumerate(f, 1):
            pass
        return i


if __name__ == '__main__':
    path = Path.cwd().parent.joinpath('cycles', 'input', 'ContinuousCorn.operation')
    parser = OperationManager(path)
    print(parser)
    parser.sort_operation()
    print(parser)

    path = Path.cwd().parent.joinpath('cycles', 'input', 'RockSprings.weather')
    weather = WeatherManager(path)
    print(weather.mutables.info())
    print(weather.immutables.info())

    path = Path.cwd().parent.joinpath('cycles', 'input', 'ContinuousCorn.ctrl')
    ctrl = ControlManager(path)
    print(ctrl)

    path = Path.cwd().parent.joinpath('cycles', 'output', 'ContinuousCorn', 'CornRM.90.dat')
    crop = CropManager(path)
    print(crop.crop_state.head())
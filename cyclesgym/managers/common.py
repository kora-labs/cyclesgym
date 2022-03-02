from pathlib import Path
from abc import ABC, abstractmethod


class Manager(ABC):
    def __init__(self, fname=None):
        self.fname = fname
        if not self._valid_input_file(self.fname):
            raise ValueError('File not existing. To initialize manager without a file, pass None as input')
        self._parse()

    @staticmethod
    def _valid_input_file(fname):
        if fname is not None:
            return fname.is_file()
        else:
            return True

    @abstractmethod
    def _parse(self):
        raise NotImplementedError

    def update_file(self, fname):
        self.fname = fname
        if not self._valid_input_file(self.fname):
            raise ValueError(f'{self.fname} File not existing. To initialize manager without a file, pass None as input')
        self._parse()


class InputFileManager(Manager):

    @staticmethod
    def _valid_output_file(fname):
        return fname is not None

    @abstractmethod
    def _to_str(self):
        raise NotImplementedError

    def __str__(self):
        return self._to_str()

    def save(self, fname, force=True):
        if not force and fname.is_file():
            raise RuntimeError(f'The file {fname} already exists. Use force=True if you want to overwrite it')
        if not self._valid_output_file(fname):
            raise ValueError(f'{fname} is not a valid path to save the file')
        else:
            s = self._to_str()
            with open(fname, 'w') as fp:
                fp.write(s)


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
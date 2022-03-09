from cyclesgym.managers.common import InputFileManager


__all__ = ['ControlManager']


class ControlManager(InputFileManager):
    def __init__(self, fname=None):
        self.ctrl_dict = dict()
        self._non_numeric = ['CROP_FILE', 'OPERATION_FILE', 'SOIL_FILE', 'WEATHER_FILE', 'REINIT_FILE']
        super().__init__(fname)

    def _parse(self, fname):
        if fname is not None:
            with open(fname, 'r') as f:
                for line in f.readlines():
                    if line.startswith(('\n', '#')):
                        pass
                    else:
                        k, v = line.split()[0:2]
                        v = v if k in self._non_numeric else int(v)
                        self.ctrl_dict.update({k: v})

    def _to_str(self):
        s = ''
        for k, v in self.ctrl_dict.items():
            s += f"{k:30}{v}\n"
            if k.startswith('ROTATION_SIZE'):
                s += '\n\n\n'
            elif k.startswith('ANNUAL_NFLUX_OUT'):
                s += '\n\n\n'
        return s
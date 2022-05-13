from cyclesgym.managers.common import InputFileManager
from copy import copy, deepcopy


__all__ = ['ReinitManager']

do_not_reinit = -999

FIRST_KEYS = ['STANRESIDUEC',
              'FLATRESIDUEC',
              'STANRESIDUEN',
              'FLATRESIDUEN',
              'MANURESURFACEC',
              'MANURESURFACEN',
              'STANRESIDUEWATER',
              'FLATRESIDUEWATER',
              'INFILTRATION']

SECOND_KEYS = ['LAYER',
               'SMC',
               'NO3',
               'NH4',
               'SOC',
               'SON',
               'MBC',
               'MBN',
               'RESABGDC',
               'RESRTC',
               'RESRZC',
               'RESIDUEABGDN',
               'RESIDUERTN',
               'RESIDUERZN',
               'MANUREC',
               'MANUREN',
               'SATURATION']


class ReinitManager(InputFileManager):
    _keys = ['']

    _non_numeric = _keys[-5:]

    def __init__(self, fname=None):
        self.ctrl_dict = dict()
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

    @classmethod
    def from_dict(cls, d):
        l1 = list(set(d.keys()) - set(cls._keys))
        l2 = list(set(cls._keys) - set(d.keys()))
        if len(l1) > 0:
            raise ValueError(
                f'The keys {l1} are not needed when specifying a {cls.__name__} from a dictionary')
        if len(l2) > 0:
            raise ValueError(
                f'The keys {l2} are necessary when specifying a {cls.__name__} from a dictionary')
        manager = cls(fname=None)
        manager.ctrl_dict = d.copy()
        return manager

    def __copy__(self):
        cls = self.__class__
        result = cls.from_dict(copy(self.ctrl_dict))
        return result

    def __deepcopy__(self):
        cls = self.__class__
        result = cls.from_dict(deepcopy(self.ctrl_dict))
        return result

import os as _os
import weakref
import numpy as np
import datetime
from uuid import uuid4

from pathlib import Path
from tempfile import TemporaryDirectory


__all__ = ['MyTemporaryDirectory', 'create_sim_id', 'date2ydoy', 'ydoy2date',
           'cap_date']


class MyTemporaryDirectory(TemporaryDirectory):
    """
    Subclass of temporary directory with specific name rather than randomly
    generated.
    """

    def __init__(self, path):
        assert isinstance(path, Path)
        _os.mkdir(path, 0o700)
        self.name = path
        self._finalizer = weakref.finalize(
            self, self._cleanup, self.name,
            warn_message="Implicitly cleaning up {!r}".format(self))


def create_sim_id():
    return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S-') + str(uuid4())


def date2ydoy(date):
    def date2ydoy_single(date: datetime.date):
        tmp = date.timetuple()
        return tmp.tm_year, tmp.tm_yday

    if hasattr(date, '__iter__'):
        y = []
        doy = []
        for d in date:
            y_tmp, doy_tmp = date2ydoy_single(d)
            y.append(y_tmp)
            doy.append(doy_tmp)
        return y, doy

    else:
        return date2ydoy_single(date)


def ydoy2date(y, doy):
    def ydoy2date_single(y: int, doy: int):
        # Timedelta does not work with np.int64
        if isinstance(y, np.int64):
            y = int(y)
        if isinstance(doy, np.int64):
            doy = int(doy)
        return datetime.date(y, 1, 1) + datetime.timedelta(doy - 1)

    if hasattr(y, '__iter__') and hasattr(doy, '__iter__'):
        if len(y) == len(doy):
            dates = []
            for y_tmp, doy_tmp in zip(y, doy):
                dates.append(ydoy2date_single(y_tmp, doy_tmp))
            return dates
        else:
            raise ValueError(f'year and doy list should have the same length. '
                             f'They have {len(y)} and {len(doy)} lengths '
                             f'instead.')
    else:
        return ydoy2date_single(y, doy)


def cap_date(date: datetime.date, end_year: int):
    """
    Clip date to final year.
    """
    return min([date,
                datetime.date(year=end_year, month=12, day=31)])

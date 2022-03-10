import os as _os
import weakref

from datetime import datetime
from uuid import uuid4
from tempfile import TemporaryDirectory


__all__ = ['MyTemporaryDirectory', 'create_sim_id']


class MyTemporaryDirectory(TemporaryDirectory):
    """
    Subclass of temporary directory with specific name rather than randomly
    generated.
    """

    def __init__(self, path):
        _os.mkdir(path, 0o700)
        self.name = path
        self._finalizer = weakref.finalize(
            self, self._cleanup, self.name,
            warn_message="Implicitly cleaning up {!r}".format(self))


def create_sim_id():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S-') + str(uuid4())

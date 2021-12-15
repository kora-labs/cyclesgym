import numpy as np
from itertools import cycle


class OpenLoopPolicy(object):
    """
    Dummy policy that cycles through a sequence of actions
    """
    def __init__(self, action_sequence):
        self.actions = cycle(action_sequence)

    def predict(self, obs, state=None, deterministic=None):
        obs = np.asarray(obs)
        if obs.ndim == 1:
            return next(self.actions), None
        else:
            return np.atleast_1d(next(self.actions)), None

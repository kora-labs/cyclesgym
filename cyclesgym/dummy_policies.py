import numpy as np
from itertools import cycle
from stable_baselines3.common.policies import BasePolicy


class OpenLoopPolicy(BasePolicy):
    """
    Dummy policy that cycles through a sequence of actions
    """
    def __init__(self, action_sequence):
        self.actions = cycle(action_sequence)

    def forward(self, *args, **kwargs):
        pass

    def _predict(self, observation, deterministic=False):
        pass

    def predict(self, observation, state=None, episode_start=None,
                deterministic=False):
        obs = np.asarray(observation)
        if obs.ndim == 1:
            return next(self.actions), None
        else:
            return np.atleast_1d(next(self.actions)), None

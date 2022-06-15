import numpy as np
from itertools import cycle
from stable_baselines3.common.policies import BasePolicy
from abc import ABC, abstractmethod


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


class ActionProcesser(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def process(self, action):
        pass


class ActionBinner(ActionProcesser):
    def __init__(self, n_bins, lower, upper):
        self.n_bins = np.array(n_bins)
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        assert self.n_bins.size == self.lower.size == self.upper.size

    def process(self, action):
        processed_action = np.zeros_like(self.lower).astype(int)
        for i, (n, l, u, a) in enumerate(zip(self.n_bins, self.lower, self.upper, action)):
            a = np.clip(a, l, u)
            processed_action[i] = np.digitize(a, np.linspace(l, u, n)) - 1
        if processed_action.size == 1:
            return processed_action[0]
        else:
            return processed_action


class LinearPolicy(BasePolicy):
    def __init__(self, K, action_post_processing):
        self.K = K
        self.action_post_processing = action_post_processing

    def forward(self, *args, **kwargs):
        pass

    def _predict(self, observation, deterministic=False):
        pass

    def predict(self, observation, state=None, episode_start=None,
                deterministic=False):
        obs = np.asarray(observation)
        raw_action = np.matmul(self.K, obs)
        return self.action_post_processing.process(raw_action), None

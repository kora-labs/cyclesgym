from cyclesgym.informed_policy import InformedPolicy
from pathlib import Path
import unittest
import gym
from gym import spaces
import numpy as np
from numpy.testing import *


class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=np.array([0,0]),
                                            high=np.array([365,120]), dtype=int)
        self.action_space = spaces.Discrete(3)
        self.state = self.observation_space.sample()
        self.n_steps = 0
        self.maxN = 100
        self.n_actions = 3

    def step(self, action):
        self.state += np.random.choice([-1, 1]) * action
        self.n_steps += 1
        return self.state, np.random.rand(1), self.n_steps > 5, {}

    def reset(self):
        self.state = self.observation_space.sample()
        self.n_steps = 0
        return self.state

    def render(self, mode="human"):
        pass


class TestInformedPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.env = DummyEnv()
        params = (100, # start_day
                  200, # end_day
                  20, # a
                  1.0, # max_val
                  0.0, # min_val
                  100, # saturation
                  100 # lengthscale
                  )
        self.model = InformedPolicy(env=self.env, params=params)
        self.obs = np.array([[50, 0],    # Before window starts
                            [250, 40],  # After window is over
                            [110, 50],  # Middle of down ramp
                            [150, 50]   # Middle of window
                            ])

    def test_action_prob(self):
        probs = self.model.action_probability(self.obs)

        Z = 2*np.exp(-50**2 / 100) + 1
        target_probs = np.array([[1, 0, 0],
                                 [1, 0, 0],
                                 [0.5 + (0.5 * np.exp(-50**2 / 100))/Z, 0.5/Z, (0.5 * np.exp(-50**2 / 100))/Z],
                                 [np.exp(-50**2 / 100)/Z, 1/Z, np.exp(-50**2 / 100)/Z]])
        assert_allclose(probs, target_probs)

    def test_predict(self):
        # Test multiple deterministic
        actions, _ = self.model.predict(self.obs, deterministic=True)
        target_actions = np.array([0, 0, 0, 1])
        assert_equal(actions, target_actions)

        # Test single deterministic
        actions, _ = self.model.predict(self.obs[-1, :], deterministic=True)
        target_actions = np.array([1])
        assert_equal(actions, target_actions)

        # Test multiple stochastic (probability is so peaked that results is same as deterministic)
        actions, _ = self.model.predict(self.obs[[0, 3], :], deterministic=False)
        target_actions = np.array([0, 1])
        assert_equal(actions, target_actions)

        # Test single stochastic
        actions, _ = self.model.predict(self.obs[-1, :], deterministic=False)
        target_actions = np.array([1])
        assert_equal(actions, target_actions)


if __name__ == '__main__':
    unittest.main()
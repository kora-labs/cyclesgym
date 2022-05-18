from cyclesgym.envs.corn_multiyear import _CornMultiYearContinue, CornMultiYear
from cyclesgym.utils import compare_env, maximum_absolute_percentage_error
import unittest


class TestCornEnv(unittest.TestCase):

    def setUp(self) -> None:
        delta = 7
        n_actions = 7
        maxN = 120
        self.START_YEAR = 1980
        self.END_YEAR = 1983

        self.env_cont = _CornMultiYearContinue(self.START_YEAR, self.END_YEAR, delta=delta,
                                               n_actions=n_actions,
                                               maxN=maxN)

        self.env_impr = CornMultiYear(self.START_YEAR, self.END_YEAR, delta=delta,
                                      n_actions=n_actions,
                                      maxN=maxN)

    def test_fast_multiyear_against_continuous_multiyear(self):
        obs_cont, obs_impr, time_cont, time_impr = compare_env(self.env_cont, self.env_impr)
        print(f'Time of continuous environemnt over {self.END_YEAR - self.START_YEAR} years: {time_cont}')
        print(f'Time of improved environemnt over {self.END_YEAR - self.START_YEAR} years: {time_impr}')

        max_ape = maximum_absolute_percentage_error(obs_cont, obs_impr)
        print(f'Maximum percentage error: {max_ape} %')
        self.assertLess(max_ape, 1, f'Maximum percentage error: {max_ape} % is less then the threshold 1')

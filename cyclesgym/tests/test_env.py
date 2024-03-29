import pathlib
import shutil
import unittest
import numpy as np
import subprocess

from cyclesgym.envs.corn import Corn
from cyclesgym.envs.common import PartialObsEnv
from cyclesgym.envs.utils import date2ydoy
from cyclesgym.managers import *
from cyclesgym.utils.utils import compare_env, maximum_absolute_percentage_error
from cyclesgym.utils.paths import CYCLES_PATH, TEST_PATH

TEST_FILENAMES = ['NCornTest.ctrl',
                  'NCornTest.operation',
                  'NCornTestNoFertilization.operation']


def copy_cycles_test_files():
    # Copy test files in cycles input directory
    for n in TEST_FILENAMES:
        shutil.copy(TEST_PATH.joinpath(n),
                    CYCLES_PATH.joinpath('input', n))


def remove_cycles_test_files():
    # Remove all files copied to input folder
    for n in TEST_FILENAMES:
        pathlib.Path(CYCLES_PATH.joinpath('input', n)).unlink()
    # Remove the output of simulation started manually
    try:
        shutil.rmtree(pathlib.Path(CYCLES_PATH.joinpath(
            'output', TEST_FILENAMES[0].replace('.ctrl', ''))))
    except FileNotFoundError:
        pass


class TestCornEnv(unittest.TestCase):
    def setUp(self):
        copy_cycles_test_files()

    def tearDown(self):
        remove_cycles_test_files()

    def test_manual_sim_comparison(self):
        """
        Check the manual and env simulation are the same for same management
        and different for different management.
        """
        # Start normal simulation and parse results
        base_ctrl_man = TEST_FILENAMES[0].replace('.ctrl', '')
        self._call_cycles(base_ctrl_man)
        crop_man = CropManager(
            CYCLES_PATH.joinpath('output', base_ctrl_man, 'CornRM.90.dat'))
        season_man = SeasonManager(
            CYCLES_PATH.joinpath('output', base_ctrl_man, 'season.dat'))

        # Start gym env
        operation_file = TEST_FILENAMES[2]
        env = Corn(delta=1, maxN=150, n_actions=16,
                      operation_file=operation_file)

        # Run simulation with same management and compare
        env.reset()
        while True:
            _, doy = date2ydoy(env.date)
            a = 15 if doy == 106 else 0
            _, _, done, _ = env.step(a)
            if done:
                break

        # Check crop
        crop_output_file = env._get_output_dir().joinpath('CornRM.90.dat')
        crop_env = CropManager(crop_output_file)
        assert crop_env.crop_state.equals(crop_man.crop_state)

        # Check yield
        season_output_file = env._get_output_dir().joinpath('season.dat')
        season_env = SeasonManager(season_output_file)
        assert season_env.season_df.equals(season_man.season_df)

        # Run simulation with different management and compare
        env.reset()
        while True:
            _, doy = date2ydoy(env.date)
            a = 15 if doy == 107 else 0
            _, _, done, _ = env.step(a)
            if done:
                break
        crop_output_file = env._get_output_dir().joinpath('CornRM.90.dat')
        crop_env = CropManager(crop_output_file)
        assert not crop_env.crop_state.equals(crop_man.crop_state)

        season_output_file = env._get_output_dir().joinpath('season.dat')
        season_env = SeasonManager(season_output_file)
        assert not season_env.season_df.equals(season_man.season_df)

    def test_reward(self):
        # Should test reward in no fertilization case to avoid old bug
        pass

    def test_fast_multiyear_against_continuous_multiyear(self):
        self.START_YEAR = 1980
        self.END_YEAR = 1983

        delta = 7
        n_actions = 7
        maxN = 120

        self.env_cont = Corn(delta=delta, n_actions=n_actions, maxN=maxN, start_year=self.START_YEAR,
                             end_year=self.END_YEAR, use_reinit=False)

        self.env_impr = Corn(delta=delta, n_actions=n_actions, maxN=maxN, start_year=self.START_YEAR,
                             end_year=self.END_YEAR)

        obs_cont, obs_impr, time_cont, time_impr = compare_env(self.env_cont, self.env_impr)
        print(f'Time of continuous environemnt over {self.END_YEAR - self.START_YEAR} years: {time_cont}')
        print(f'Time of improved environemnt over {self.END_YEAR - self.START_YEAR} years: {time_impr}')

        max_ape = maximum_absolute_percentage_error(obs_cont, obs_impr)
        print(f'Maximum percentage error: {max_ape} %')
        self.assertLess(max_ape, 1, f'Maximum percentage error: {max_ape} % is less then the threshold 1')

    @staticmethod
    def _call_cycles(ctrl):
        subprocess.run(['./Cycles', '-b', ctrl], cwd=CYCLES_PATH)


class TestPartiallyObservableEnv(unittest.TestCase):
    def setUp(self) -> None:
        copy_cycles_test_files()
        self.f_env = Corn(delta=1, maxN=150, n_actions=16,
                                operation_file=TEST_FILENAMES[2])
        self.base_env = Corn(delta=1, maxN=150, n_actions=16,
                                operation_file=TEST_FILENAMES[2])
        n_obs = np.prod(self.base_env.observation_space.shape)

        np.random.seed(0)
        self.mask = np.random.choice(2, size=n_obs).astype(bool)
        self.wrapped_env = PartialObsEnv(self.base_env, mask=self.mask)

    def tearDown(self) -> None:
        remove_cycles_test_files()

    def test_full_vs_partial_observation(self):
        f_obs = self.f_env.reset()
        p_obs = self.wrapped_env.reset()
        print(self.wrapped_env.env.observer.obs_names)

        assert np.all(p_obs == f_obs[self.mask])
        for _ in range(5):
            a = self.f_env.action_space.sample()
            f_obs, f_r, f_done, _ = self.f_env.step(a)
            p_obs, p_r, p_done, _ = self.wrapped_env.step(a)
            assert np.all(p_obs == f_obs[self.mask])
            assert f_r == p_r
            assert f_done == p_done


if __name__ == '__main__':
    unittest.main()

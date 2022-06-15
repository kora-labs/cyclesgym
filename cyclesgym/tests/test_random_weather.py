import pathlib
import shutil
import subprocess
import unittest
import unittest.mock as mock
import numpy as np
from cyclesgym.envs.corn import Corn, CornShuffledWeather
from cyclesgym.envs.utils import date2ydoy
from cyclesgym.managers import WeatherManager, CropManager, SeasonManager
from cyclesgym.envs.weather_generator import WeatherShuffler, generate_random_weather
from cyclesgym.paths import CYCLES_PATH, TEST_PATH

TEST_FILENAMES = ['CornRandomWeatherTest.ctrl',
                  'NCornTest.operation']


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


def fixed_start_year(valid_start_years):
    return valid_start_years[0]


def fixed_perm(sampled_years):
    copy = sampled_years.copy()
    for j, i in enumerate([2, 0, 1]):
        sampled_years[j] = copy[i]


class TestShuffleWeather(unittest.TestCase):

    def setUp(self):
        copy_cycles_test_files()

    def tearDown(self):
        remove_cycles_test_files()

    def test_shuffling_dummy_weather(self):
        fname = TEST_PATH.joinpath('DummyWeatherNonShuffle.weather')
        fname_shuffled = TEST_PATH.joinpath('DummyWeatherShuffle.weather')

        shuffler = WeatherShuffler(n_weather_samples=1,
                                   sampling_start_year=1981,
                                   sampling_end_year=1983,
                                   base_weather_file=fname,
                                   target_year_range=[1981, 1982, 1983])

        shuffled_manager = WeatherManager(fname_shuffled)

        # Create weather
        with mock.patch('numpy.random.choice', fixed_start_year), mock.patch('random.shuffle', fixed_perm):
            shuffler.generate_weather()
            new_weather = WeatherManager(shuffler._get_weather_dir().joinpath(shuffler.weather_list[0])).mutables

        assert shuffled_manager.mutables.equals(new_weather)

    def test_env_against_cycles_with_shuffled_weather(self):
        """
        Check the manual and env simulation are the same for same management
        and different for different management.
        """
        # Start normal simulation and parse results
        base_ctrl_man = TEST_FILENAMES[0].replace('.ctrl', '')
        shuffler = WeatherShuffler(n_weather_samples=1,
                                   sampling_start_year=1980,
                                   sampling_end_year=1982,
                                   base_weather_file=CYCLES_PATH.joinpath('input/RockSprings.weather'),
                                   target_year_range=[1980, 1981, 1982])
        with mock.patch('numpy.random.choice', fixed_start_year), mock.patch('random.shuffle', fixed_perm):
            shuffler.generate_weather()

        weather_file = shuffler._get_weather_dir().joinpath(shuffler.weather_list[0])
        shutil.copy(weather_file,
                    CYCLES_PATH.joinpath('input', shuffler.weather_list[0]))

        self._call_cycles(base_ctrl_man)

        pathlib.Path(CYCLES_PATH.joinpath(
            'input', shuffler.weather_list[0])).unlink()
        crop_man = CropManager(
            CYCLES_PATH.joinpath('output', base_ctrl_man, 'CornRM.90.dat'))
        season_man = SeasonManager(
            CYCLES_PATH.joinpath('output', base_ctrl_man, 'season.dat'))

        # Start gym env
        operation_file = TEST_FILENAMES[1]
        with mock.patch('numpy.random.choice', fixed_start_year), mock.patch('random.shuffle', fixed_perm):
            env = CornShuffledWeather(delta=1, maxN=150, n_actions=16,
                                      start_year=1980,
                                      end_year=1982,
                                      use_reinit=False,
                                      n_weather_samples=1,
                                      operation_file=operation_file,
                                      sampling_start_year=1980,
                                      sampling_end_year=1982
                                      )

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

    @staticmethod
    def _call_cycles(ctrl):
        subprocess.run(['./Cycles', '-b', ctrl], cwd=CYCLES_PATH)
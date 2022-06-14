import pathlib
import shutil
import subprocess
import unittest
import unittest.mock as mock
import numpy as np
from cyclesgym.envs.corn import Corn
from cyclesgym.envs.utils import date2ydoy
from cyclesgym.managers import WeatherManager
from cyclesgym.envs.weather_generator import WeatherShuffler, generate_random_weather
from cyclesgym.paths import CYCLES_PATH, TEST_PATH


class TestShuffleWeather(unittest.TestCase):
    def setUp(self):
        copy_cycles_test_files()

    def tearDown(self):
        remove_cycles_test_files()

    def test_shuffling_dummy_weather(self):
        fname = TEST_PATH.joinpath('DummyWeatherNonShuffle.weather')
        fname_shuffled = TEST_PATH.joinpath('DummyWeatherShuffle.weather')

        def fixed_start_year(valid_start_years):
            return valid_start_years[0]

        def fixed_perm(sampled_years):
            copy = sampled_years.copy()
            for j, i in enumerate([2, 0, 1]):
                sampled_years[j] = copy[i]

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
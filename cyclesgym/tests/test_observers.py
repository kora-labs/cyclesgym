import datetime
import unittest
import numpy as np

from cyclesgym.envs.observers import WheatherObserver, CropObserver
from cyclesgym.managers import *

from cyclesgym.paths import TEST_PATH


class TestObservers(unittest.TestCase):
    def setUp(self) -> None:
        # Init managers
        self.weather_manager = WeatherManager(
            TEST_PATH.joinpath('DummyWeather.weather'))
        self.crop_manager = CropManager(
            TEST_PATH.joinpath('DummyCrop.dat'))

        # Init observers
        self.weather_observer = WheatherObserver(
            weather_manager=self.weather_manager,
            end_year=1980
        )
        self.crop_observer = CropObserver(
            crop_manager=self.crop_manager,
            end_year=1980
        )

    def test_weather_obs(self):
        date = datetime.date(year=1980, month=1, day=1)
        obs = self.weather_observer.compute_obs(date)
        target_obs = np.concatenate((np.array([40.6875, 0, 10]), np.arange(7)))
        assert np.all(obs == target_obs)

    def test_crop_obs(self):
        date = datetime.date(year=1980, month=1, day=1)
        obs = self.crop_observer.compute_obs(date)
        target_obs = np.arange(14)
        assert np.all(obs == target_obs)


if __name__ == '__main__':
    unittest.main()

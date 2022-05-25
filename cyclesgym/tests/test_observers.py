import datetime
import unittest
import numpy as np

from cyclesgym.envs.observers import WeatherObserver, CropObserver, compound_observer, NToDateObserver, SoilNObserver
from cyclesgym.managers import *

from cyclesgym.paths import TEST_PATH


class TestObservers(unittest.TestCase):
    def setUp(self) -> None:
        # Init managers
        self.weather_manager = WeatherManager(
            TEST_PATH.joinpath('DummyWeather.weather'))
        self.crop_manager = CropManager(
            TEST_PATH.joinpath('DummyCrop.dat'))
        self.soil_manager = SoilNManager(
            TEST_PATH.joinpath('DummyN.dat'))

        # Init observers
        self.weather_observer = WeatherObserver(
            weather_manager=self.weather_manager,
            end_year=1980
        )
        self.crop_observer = CropObserver(
            crop_manager=self.crop_manager,
            end_year=1980
        )
        self.soil_observer = SoilNObserver(
            soil_n_manager=self.soil_manager,
            end_year=1980
        )
        self.N_to_date_observer = NToDateObserver(end_year=1980)
        self.weather_crop_observer = compound_observer([self.weather_observer, self.crop_observer])
        self.weather_target_obs = np.concatenate((np.array([40.6875, 0, 10]), np.arange(7)))
        self.crop_target_obs = np.concatenate(([0], np.arange(14)))
        self.N_to_date_target_obs = list(zip(np.arange(1, 11), np.arange(2, 21, step=2)))
        self.N_to_date_target_obs =[np.array(ele) for ele in self.N_to_date_target_obs]
        self.soil_target_obs = np.array([8873.886558, 37.068387, 8.968119, 0.037371, 0.000000, 0.037371, 0.068560,
                                         0.00000, 0.00000, 0.000000, 0.000000])

    def test_weather_obs(self):
        date = datetime.date(year=1980, month=1, day=1)
        obs = self.weather_observer.compute_obs(date)
        assert np.all(obs == self.weather_target_obs)

    def test_crop_obs(self):
        date = datetime.date(year=1980, month=1, day=1)
        obs = self.crop_observer.compute_obs(date)
        assert np.all(obs == self.crop_target_obs)

    def test_compound_observer(self):
        date = datetime.date(year=1980, month=1, day=1)
        obs = self.weather_crop_observer.compute_obs(date)
        assert np.all(obs == np.concatenate([self.weather_target_obs, self.crop_target_obs]))
        assert self.weather_crop_observer.Nobs == 25

    def test_N_to_date_observer(self):
        date = datetime.date(year=1980, month=1, day=1)
        obs = []
        for i in range(10):
            obs.append(self.N_to_date_observer.compute_obs(date, N=2))
            date = date + datetime.timedelta(days=1)

        assert np.all(np.all(o == target for o, target in zip(obs, self.N_to_date_target_obs)))

    def test_soil_N_observer(self):
        date = datetime.date(year=1980, month=1, day=1)

        obs = self.soil_observer.compute_obs(date)
        assert np.all(obs == self.soil_target_obs)


if __name__ == '__main__':
    unittest.main()

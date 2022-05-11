import pandas as pd
import numpy as np
from cyclesgym.envs.utils import date2ydoy
import datetime
from cyclesgym.managers import WeatherManager, CropManager

__all__ = ['WeatherCropObserver', 'WeatherCropDoyNObserver']


class WeatherCropObserver(object):
    Nobs = 24
    lower_bound = np.full((Nobs,), -np.inf)
    upper_bound = np.full((Nobs,), np.inf)

    def __init__(self,
                 weather_manager: WeatherManager,
                 crop_manager: CropManager,
                 end_year: int):
        self.weather_manager = weather_manager
        self.crop_manager = crop_manager
        self.obs_names = None
        self.end_year = end_year

    def compute_obs(self,
                    date: datetime.date):
        # Make sure we did not go into not simulated year when advancing time
        date = min([date,
                    datetime.date(year=self.end_year, month=12, day=31)])
        year, doy = date2ydoy(date)

        crop_data = self.crop_manager.get_day(year, doy).iloc[0, 4:]
        imm_weather_data = self.weather_manager.immutables.iloc[0, :]
        mutable_weather_data = self.weather_manager.get_day(year, doy).iloc[0,
                               2:]

        obs = pd.concat([crop_data, imm_weather_data, mutable_weather_data])
        if self.obs_names is None:
            self.obs_names = list(obs.index)

        return obs.to_numpy(dtype=float)


class WeatherCropDoyNObserver(WeatherCropObserver):
    Nobs = 26
    lower_bound = np.full((Nobs,), -np.inf)
    upper_bound = np.full((Nobs,), np.inf)

    def __init__(self,
                 weather_manager: WeatherManager,
                 crop_manager: CropManager,
                 end_year: int):
        self.N_to_date = 0
        super().__init__(weather_manager, crop_manager, end_year)

    # TODO: Fix Liskov substitution principle (same signature as parent method)
    def compute_obs(self,
                    date: datetime.date,
                    N: float):
        obs = super().compute_obs(date)
        if self.obs_names[-1] != 'N TO DATE':
            self.obs_names += ['DOY', 'N TO DATE']

        _, doy = date2ydoy(date)

        self.N_to_date += N
        return np.append(obs, [doy, self.N_to_date])

    def reset(self):
        self.N_to_date = 0



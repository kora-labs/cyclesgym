import pandas as pd
import numpy as np
from cyclesgym.envs.utils import date2ydoy
import datetime
from cyclesgym.managers import WeatherManager, CropManager, SoilNManager

__all__ = ['WeatherCropObserver', 'WeatherCropDoyNObserver', 'WheatherObserver', 'CropObserver', 'compound_observer']


class WheatherObserver(object):
    Nobs = 24
    lower_bound = np.full((Nobs,), -np.inf)
    upper_bound = np.full((Nobs,), np.inf)

    def __init__(self,
                 weather_manager: WeatherManager,
                 end_year: int):
        self.weather_manager = weather_manager
        self.obs_names = None
        self.end_year = end_year

    def compute_obs(self,
                    date: datetime.date):
        # Make sure we did not go into not simulated year when advancing time
        date = min([date,
                    datetime.date(year=self.end_year, month=12, day=31)])
        year, doy = date2ydoy(date)

        imm_weather_data = self.weather_manager.immutables.iloc[0, :]
        mutable_weather_data = self.weather_manager.get_day(year, doy).iloc[0,
                               2:]

        obs = pd.concat([imm_weather_data, mutable_weather_data])
        if self.obs_names is None:
            self.obs_names = list(obs.index)

        return obs.to_numpy(dtype=float)


class CropObserver(object):
    Nobs = 24
    lower_bound = np.full((Nobs,), -np.inf)
    upper_bound = np.full((Nobs,), np.inf)

    def __init__(self,
                 crop_manager: CropManager,
                 end_year: int):
        self.crop_manager = crop_manager
        self.obs_names = None
        self.end_year = end_year

    def compute_obs(self,
                    date: datetime.date):
        # Make sure we did not go into not simulated year when advancing time
        date = min([date,
                    datetime.date(year=self.end_year, month=12, day=31)])
        year, doy = date2ydoy(date)

        crop_data = self.crop_manager.get_day(year, doy)
        if not crop_data.empty:
            crop_data = crop_data.iloc[0, 4:]

        obs = crop_data
        if self.obs_names is None:
            self.obs_names = list(obs.index)

        return obs.to_numpy(dtype=float)


# TODO: to be written as compounded Wheather and Crop observer
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


# TODO: to be written as compounded Whether Crop and DoyN
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


# TODO: write generic daily observer, that is a parent of SoilN, wheather, Crop
class SoilNObserver(CropObserver):
    Nobs = 15
    lower_bound = np.full((Nobs,), -np.inf)
    upper_bound = np.full((Nobs,), np.inf)

    def __init__(self,
                 soil_n_manager: SoilNManager,
                 end_year: int):
        self.soil_n_manager = soil_n_manager
        self.obs_names = None
        self.end_year = end_year

    def compute_obs(self,
                    date: datetime.date):
        # Make sure we did not go into not simulated year when advancing time
        date = min([date,
                    datetime.date(year=self.end_year, month=12, day=31)])
        year, doy = date2ydoy(date)

        soil_n_data = self.soil_n_manager.get_day(year, doy)
        if not soil_n_data.empty:
            soil_n_data = soil_n_data.iloc[0, 2:]

        obs = soil_n_data
        if self.obs_names is None:
            self.obs_names = list(obs.index)

        return obs.to_numpy(dtype=float)


def compound_observer(obs_list: list):

    class Compound(object):
        def __init__(self, obs_list):
            self.obs_list = obs_list

        def compute_obs(self, date: datetime.date):
            obs = [o.compute_obs(date).squeeze() for o in self.obs_list]
            obs = [o for o in obs if o.size > 0]
            self.obs_names = [name for o in obs_list for name in o.obs_names]

            return np.concatenate(obs)

    return Compound(obs_list)



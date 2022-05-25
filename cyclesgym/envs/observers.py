import pandas as pd
import numpy as np
from cyclesgym.envs.utils import date2ydoy, cap_date
import datetime
from cyclesgym.managers import WeatherManager, CropManager, SoilNManager

__all__ = ['WeatherObserver', 'CropObserver', 'SoilNObserver', 'compound_observer']


class Observer(object):

    def __init__(self, end_year: int):
        self.end_year = end_year
        self.obs_names = None
        self.Nobs = 0
        self.lower_bound = None
        self.upper_bound = None

    def compute_obs(self, date: datetime.date, **kwargs):
        raise NotImplementedError


class WeatherObserver(Observer):

    def __init__(self,
                 weather_manager: WeatherManager,
                 end_year: int):
        super(WeatherObserver, self).__init__(end_year)
        self.weather_manager = weather_manager
        self.Nobs = 10
        self.lower_bound = np.full((self.Nobs,), -np.inf)
        self.upper_bound = np.full((self.Nobs,), np.inf)

    def compute_obs(self,
                    date: datetime.date,
                    **kwargs):
        # Make sure we did not go into not simulated year when advancing time
        date = cap_date(date, self.end_year)
        year, doy = date2ydoy(date)

        imm_weather_data = self.weather_manager.immutables.iloc[0, :]
        mutable_weather_data = self.weather_manager.get_day(year, doy).iloc[0, 2:]

        obs = pd.concat([imm_weather_data, mutable_weather_data])
        if self.obs_names is None:
            self.obs_names = list(obs.index)

        self.Nobs = obs.size
        return obs.to_numpy(dtype=float)


class DailyOutputObserver(Observer):

    def __init__(self, manager, end_year: int):
        super(DailyOutputObserver, self).__init__(end_year)
        self.manager = manager
        self.observed_columns = None

    def compute_obs(self,
                    date: datetime.date,
                    **kwargs):
        # Make sure we did not go into not simulated year when advancing time
        date = cap_date(date, self.end_year)
        year, doy = date2ydoy(date)

        data = self.manager.get_day(year, doy)
        if not data.empty:
            data = data.iloc[0, self.observed_columns]

        obs = data
        if self.obs_names is None:
            self.obs_names = list(obs.index)

        self.Nobs = obs.size
        return obs.to_numpy(dtype=float)


class CropObserver(DailyOutputObserver):

    def __init__(self,
                 crop_manager: CropManager,
                 end_year: int):
        super(CropObserver, self).__init__(crop_manager, end_year)
        self.Nobs = 14
        self.lower_bound = np.full((self.Nobs,), -np.inf)
        self.upper_bound = np.full((self.Nobs,), np.inf)
        self.observed_columns = 4 + np.arange(self.Nobs)


class NToDateObserver(Observer):

    def __init__(self,
                 end_year: int):
        super(NToDateObserver, self).__init__(end_year)
        self.N_to_date = 0
        self.Nobs = 2
        self.lower_bound = np.full((self.Nobs,), -np.inf)
        self.upper_bound = np.full((self.Nobs,), np.inf)

    # TODO: Fix Liskov substitution principle (same signature as parent method)
    def compute_obs(self,
                    date: datetime.date,
                    N: float):

        # Make sure we did not go into not simulated year when advancing time
        date = cap_date(date, self.end_year)
        self.obs_names = ['DOY', 'N TO DATE']

        _, doy = date2ydoy(date)
        self.N_to_date += N
        return np.array([doy, self.N_to_date])

    def reset(self):
        self.N_to_date = 0


class SoilNObserver(DailyOutputObserver):

    def __init__(self,
                 soil_n_manager: SoilNManager,
                 end_year: int):
        super(SoilNObserver, self).__init__(soil_n_manager, end_year)
        self.Nobs = 11
        self.lower_bound = np.full((self.Nobs,), -np.inf)
        self.upper_bound = np.full((self.Nobs,), np.inf)
        self.observed_columns = 2 + np.arange(self.Nobs)


def compound_observer(obs_list: list):

    class Compound(object):
        def __init__(self, obs_list):
            self.obs_list = obs_list
            self.Nobs = sum([o.Nobs for o in obs_list])
            self.lower_bound = np.full((self.Nobs,), -np.inf)
            self.upper_bound = np.full((self.Nobs,), np.inf)

        def compute_obs(self, date: datetime.date, **kwargs):
            obs = [o.compute_obs(date, **kwargs).squeeze() for o in self.obs_list]
            obs = [o for o in obs if o.size > 0]
            self.obs_names = [name for o in obs_list for name in o.obs_names]

            new_Nobs = sum([o.size for o in obs])
            if new_Nobs != self.Nobs:
                print(f'Warning: runtime number of observation for {self} is different then the original'
                      f'one: Before: {self.Nobs}, runtime: {new_Nobs}')
                print(self.obs_list)
            self.Nobs = new_Nobs
            return np.concatenate(obs)

    return Compound(obs_list)



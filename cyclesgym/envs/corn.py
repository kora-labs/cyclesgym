from datetime import timedelta
from cyclesgym.envs.common import CyclesEnv
from cyclesgym.envs.observers import compound_observer, CropObserver, \
    WeatherObserver, NToDateObserver
from cyclesgym.envs.rewarders import compound_rewarder, CropRewarder, \
    NProfitabilityRewarder
from cyclesgym.envs.implementers import *

from typing import Tuple

import numpy as np
from gym import spaces


from cyclesgym.managers import *

__all__ = ['CornNew']


class CornNew(CyclesEnv):
    def __init__(self, delta,
                 n_actions,
                 maxN,
                 operation_file='ContinuousCorn.operation',
                 soil_file='GenericHagerstown.soil',
                 weather_file='RockSprings.weather'
                 ):
        super().__init__(SIMULATION_START_YEAR=1980,
                         SIMULATION_END_YEAR=1980,
                         ROTATION_SIZE=1,
                         USE_REINITIALIZATION=0,
                         ADJUSTED_YIELDS=0,
                         HOURLY_INFILTRATION=1,
                         AUTOMATIC_NITROGEN=0,
                         AUTOMATIC_PHOSPHORUS=0,
                         AUTOMATIC_SULFUR=0,
                         DAILY_WEATHER_OUT=0,
                         DAILY_CROP_OUT=1,
                         DAILY_RESIDUE_OUT=0,
                         DAILY_WATER_OUT=0,
                         DAILY_NITROGEN_OUT=0,
                         DAILY_SOIL_CARBON_OUT=0,
                         DAILY_SOIL_LYR_CN_OUT=0,
                         ANNUAL_SOIL_OUT=0,
                         ANNUAL_PROFILE_OUT=0,
                         ANNUAL_NFLUX_OUT=0,
                         CROP_FILE='GenericCrops.crop',
                         OPERATION_FILE=operation_file,
                         SOIL_FILE=soil_file,
                         WEATHER_FILE=weather_file,
                         REINIT_FILE='N / A',
                         delta=delta)
        self._post_init_setup()
        self._init_observer()
        self._generate_observation_space()
        self._generate_action_space(n_actions, maxN)

    def _generate_action_space(self, n_actions, maxN):
        self.action_space = spaces.Discrete(n_actions, )
        self.maxN = maxN
        self.n_actions = n_actions

    def _generate_observation_space(self):
        self.observation_space = spaces.Box(
            low=np.array(self.observer.lower_bound, dtype=np.float32),
            high=np.array(self.observer.upper_bound, dtype=np.float32),
            shape=self.observer.lower_bound.shape,
            dtype=np.float32)

    def _init_input_managers(self):
        self.weather_manager = WeatherManager(self.weather_input_file)
        self.input_managers = [self.weather_manager]
        self.input_files = [self.weather_input_file]

    def _init_output_managers(self):
        self.crop_output_file = self._get_output_dir().joinpath('CornRM.90.dat')
        self.season_file = self._get_output_dir().joinpath('season.dat')
        self.crop_output_manager = CropManager(self.crop_output_file)
        self.season_manager = SeasonManager(self.season_file)

        self.output_managers = [self.crop_output_manager,
                                self.season_manager]
        self.output_files = [self.crop_output_file,
                             self.season_file]

    def _init_observer(self, *args, **kwargs):
        end_year = self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR']
        self.observer = compound_observer([WeatherObserver(weather_manager=self.weather_manager, end_year=end_year),
                                           CropObserver(crop_manager=self.crop_output_manager, end_year=end_year),
                                           NToDateObserver(end_year=end_year)
                                           ])

    def _init_rewarder(self, *args, **kwargs):
        self.rewarder = compound_rewarder([CropRewarder(self.season_manager, 'CornRM.90'),
                                           NProfitabilityRewarder()])

    def _init_implementer(self, *args, **kwargs):
        self.implementer = FixedRateNFertilizer(
            operation_manager=self.op_manager,
            operation_fname=self.op_file,
            rate=0.75,
            start_year=self.ctrl_base_manager.ctrl_dict['SIMULATION_START_YEAR']
        )

    def _action2mass(self, action: int) -> float:
        return self.maxN * action / (self.n_actions - 1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        action = self._action2mass(action)
        rerun_cycles = self.implementer.implement_action(
            date=self.date, mass=action)

        if rerun_cycles:
            self._call_cycles(debug=False)

        # Advance time
        self.date += timedelta(days=self.delta)

        # Compute reward
        r = self.rewarder.compute_reward(date=self.date, delta=self.delta, action=action)

        # Compute state
        obs = self.observer.compute_obs(self.date, N=action)

        # Compute
        done = self.date.year > self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR']

        return obs, r, done, {}

    def reset(self) -> np.ndarray:
        # Set up dirs and files and run first simulation
        self._common_reset()

        # Init objects to compute obs, rewards, and implement actions
        self._init_observer()
        self._init_rewarder()
        self._init_implementer()

        # Set to zero all pre-existing fertilization for N
        self.implementer.reset()
        return self.observer.compute_obs(self.date, N=0)

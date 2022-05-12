from datetime import timedelta
from cyclesgym.envs.common import CyclesEnv
from cyclesgym.envs.corn_old import CornEnvOld
from cyclesgym.envs.observers import *
from cyclesgym.envs.rewarders import *
from cyclesgym.envs.implementers import *

from typing import Tuple

import numpy as np
from gym import spaces

from cyclesgym.managers import *

__all__ = ['CornNew']


class CornNew(CyclesEnv):
    def __init__(self, delta, n_actions, maxN):
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
                         OPERATION_FILE='ContinuousCorn.operation',
                         SOIL_FILE='GenericHagerstown.soil',
                         WEATHER_FILE='RockSprings.weather',
                         REINIT_FILE='N / A',
                         delta=delta)
        self.weather_manager = None
        self.crop_output_file = None
        self.crop_output_manager = None
        self.season_file = None
        self.season_manager = None

        # Now we can write action and obs space
        self.observation_space = spaces.Box(
            low=WeatherCropDoyNObserver.lower_bound,
            high=WeatherCropDoyNObserver.upper_bound,
            shape=WeatherCropDoyNObserver.lower_bound.shape,
            dtype=np.float32)
        self.action_space = spaces.Discrete(n_actions, )
        self.maxN = maxN
        self.n_actions = n_actions

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
        self.observer = WeatherCropDoyNObserver(
            weather_manager=self.weather_manager,
            crop_manager=self.crop_output_manager,
            end_year=self.ctrl_base_manager.ctrl_dict['SIMULATION_START_YEAR']
        )

    def _init_rewarder(self, *args, **kwargs):
        self.rewarder = CornNProfitabilityRewarder(self.season_manager)

    def _init_implementer(self, * args, **kwargs):
        self.implementer = FixedRateNFertilizer(
            operation_manager=self.op_manager,
            operation_fname=self.op_file,
            rate=0.75,
            start_year=self.ctrl_base_manager.ctrl_dict['SIMULATION_START_YEAR']
        )

    def _action2mass(self, action: int) -> float:
        return self.maxN * action / (self.n_actions - 1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        N_mass = self._action2mass(action)
        rerun_cycles = self.implementer.implement_action(
            date=self.date, mass=N_mass)

        if rerun_cycles:
            self._call_cycles(debug=False)

        # Advance time
        self.date += timedelta(days=self.delta)

        # Compute reward
        r = self.rewarder.compute_reward(
            N_mass, date=self.date, delta=self.delta)

        # Compute state
        obs = self.observer.compute_obs(self.date, N=N_mass)

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


def deploy_env(old=False):
    import time
    if old:
        env = CornEnvOld('ContinuousCorn.ctrl')
    else:
        env = CornNew(delta=7, n_actions=7, maxN=120)
    env.reset()
    t = time.time()
    while True:
        a = env.action_space.sample()
        s, r, done, info = env.step(a)
        if done:
            break
    print(f'Time elapsed:\t{time.time() - t}')


def compare_env():
    import time
    delta = 7
    n_actions = 7
    maxN=120
    old_env = CornEnvOld('ContinuousCorn.ctrl',
                         delta=delta,
                         n_actions=n_actions,
                         maxN=maxN)
    env = CornNew(delta=delta,
                  n_actions=n_actions,
                  maxN=maxN)
    s_old = old_env.reset()
    s = env.reset()
    print(f'Observation error {np.linalg.norm(s_old - s, ord=np.inf)}')

    t = time.time()
    while True:
        a = env.action_space.sample()
        s_old, r_old, done_old, info_old = old_env.step(a)
        s, r, done, info = env.step(a)
        print(f'Observation error {np.linalg.norm(s_old - s, ord=np.inf)}')
        if done:
            break
    print(f'Time elapsed:\t{time.time() - t}')


if __name__ == '__main__':
    # deploy_env(old=False)
    compare_env()

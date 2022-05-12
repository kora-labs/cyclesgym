from datetime import timedelta
from cyclesgym.envs.common import CyclesEnv
from cyclesgym.envs.corn_old import CornEnvOld
from cyclesgym.envs.observers import *
from cyclesgym.envs.rewarders import *
from cyclesgym.envs.implementers import *
from cyclesgym.envs.corn import CornNew
from typing import Tuple

import numpy as np
from gym import spaces

from cyclesgym.managers import *

__all__ = ['CornMultiYearContinue']


class CornMultiYearContinue(CornNew):

    def __init__(self, delta, n_actions, maxN):
        super(CornNew, self).__init__(SIMULATION_START_YEAR=1980,
                             SIMULATION_END_YEAR=1982,
                             ROTATION_SIZE=2,
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
        self._post_init_setup(n_actions, maxN)


class CornMultiYearSingleSteps(CornMultiYearContinue):

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        N_mass = self._action2mass(action)
        rerun_cycles = self.implementer.implement_action(
            date=self.date, mass=N_mass)

        if rerun_cycles:
            self._call_cycles(debug=False)

        # Advance time
        self.date += timedelta(days=self.delta)
        self.season_manager.update_file(self.season_file)
        # Compute reward
        r = self.rewarder.compute_reward(
            N_mass, date=self.date, delta=self.delta)

        # Compute state
        obs = self.observer.compute_obs(self.date, N=N_mass)

        # Compute
        done = self.date.year > self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR']

        return obs, r, done, {}


def compare_env():
    import time
    delta = 7
    n_actions = 7
    maxN=120
    old_env = CornMultiYearContinue(delta=delta,
                                    n_actions=n_actions,
                                    maxN=maxN)

    env = CornMultiYearSingleSteps(delta=delta,
                                   n_actions=n_actions,
                                   maxN=maxN)
    s_old = old_env.reset()
    s = env.reset()
    print(f'Observation error {np.linalg.norm(s_old - s, ord=np.inf)}')

    t = time.time()
    i = 0
    while True:
        a = env.action_space.sample()
        s_old, r_old, done_old, info_old = old_env.step(a)
        s, r, done, info = env.step(a)
        print(i)
        print(f'Observation error {np.linalg.norm(s_old - s, ord=np.inf)}')
        if done:
            break
        i = i+1
    print(f'Time elapsed:\t{time.time() - t}')


if __name__ == '__main__':
    # deploy_env(old=False)
    compare_env()

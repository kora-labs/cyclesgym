from datetime import timedelta
from cyclesgym.envs.common import CyclesEnv
from cyclesgym.envs.corn_old import CornEnvOld
from cyclesgym.envs.observers import *
from cyclesgym.envs.rewarders import *
from cyclesgym.envs.implementers import *
from cyclesgym.envs.corn import CornNew
from typing import Tuple
import shutil
import numpy as np
import pathlib
from gym import spaces

from cyclesgym.managers import *

__all__ = ['CornMultiYearContinue']

START_YEAR = 1980
END_YEAR = 1983
ROTATION_SIZE = END_YEAR - START_YEAR + 1


class CornMultiYearContinue(CornNew):

    def __init__(self, delta, n_actions, maxN):
        super(CornNew, self).__init__(SIMULATION_START_YEAR=START_YEAR,
                                      SIMULATION_END_YEAR=END_YEAR,
                                     ROTATION_SIZE=ROTATION_SIZE,
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

    def _create_operation_file(self):
        """Create operation file by copying the base one."""
        super(CornMultiYearContinue, self)._create_operation_file()
        operations = [key for key in self.op_manager.op_dict.keys() if key[2] != 'FIXED_FERTILIZATION']
        for i in range(ROTATION_SIZE-1):
            for op in operations:
                copied_op = (i+2,) + op[1:]
                self.op_manager.op_dict[copied_op] = self.op_manager.op_dict[op]

        self.op_manager.save(self.op_file)
        #TODO: write other operations only if not available. Print warining in this case.


class CornMultiYearSingleSteps(CornMultiYearContinue):

    def _create_control_file(self):
        super(CornMultiYearSingleSteps, self)._create_control_file()
        self.ctrl_manager.ctrl_dict['SIMULATION_END_YEAR'] = self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR']
        self.ctrl_manager.save(self.ctrl_file)

    def _update_control_file(self):
        self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR'] = self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR'] + 1
        self.ctrl_manager.ctrl_dict['SIMULATION_END_YEAR'] = self.ctrl_manager.ctrl_dict['SIMULATION_END_YEAR'] + 1
        self.ctrl_manager.ctrl_dict['USE_REINITIALIZATION'] = 1
        new_reinit = self.input_dir.name.joinpath('reinit.dat')
        self.ctrl_manager.ctrl_dict['REINIT_FILE'] = pathlib.Path(self.input_dir.name.stem).joinpath('reinit.dat')
        shutil.copy(self._get_output_dir().joinpath('reinit.dat'), new_reinit)

        lines = []
        with open(new_reinit, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    line = line.split()
                    line[1] = str(int(line[1]) + 1)
                    line[3] = '1'
                    line = ('    ').join(line) + '\n'
                lines.append(line)

        with open(new_reinit, 'w') as f:
            for line in lines:
                f.write(line)

        self.ctrl_manager.save(self.ctrl_file)

    def _check_is_last_step_year(self):
        return (self.date + timedelta(days=self.delta)).year != self.date.year

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        N_mass = self._action2mass(action)
        rerun_cycles = self.implementer.implement_action(
            date=self.date, mass=N_mass)

        reinit = self._check_is_last_step_year()
        doy = None
        if reinit:
            doy = 365

        if rerun_cycles:
            self._call_cycles(debug=False, reinit=reinit, doy=doy)

        # Advance time
        year = self.date.year
        self.date += timedelta(days=self.delta)
        self.changed_year = self.date.year != year
        if self.changed_year:
            self._update_control_file()

        self.season_manager.update_file(self.season_file)
        # Compute reward
        r = self.rewarder.compute_reward(
            N_mass, date=self.date, delta=self.delta)

        done = self.date.year > self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR']

        if self.changed_year and not done:
            self._call_cycles(debug=False)

        # Compute state
        obs = self.observer.compute_obs(self.date, N=N_mass)

        # Compute

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

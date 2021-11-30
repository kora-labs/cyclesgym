import shutil
from pathlib import Path
import gym
import pandas as pd
from gym import spaces
import stat
import subprocess
import time
from datetime import datetime
from uuid import uuid4
from cyclesgym.managers import *


from cyclesgym.cycles_config import CYCLES_DIR
# CYCLES_DIR = Path(__file__).parent.joinpath('cycles')


class NWinterWheatEnv(gym.Env):
    def __init__(self, ctrl, delta=7, maxN=120, n_actions=7):
        self.ctrl = ctrl.with_suffix('.ctrl')
        self.input_dir = CYCLES_DIR.joinpath('input')
        self.output_dir = CYCLES_DIR.joinpath('output', self.ctrl.stem)

        # Check control file exists and initialize managers
        if not self.input_dir.joinpath(self.ctrl).is_file():
            raise ValueError(f'There is no file named {self.ctrl} in {self.input_dir}. A valid control file is necessary'
                             f' to create an environment')
        self.ctrl_manager = ControlManager(self.input_dir.joinpath(self.ctrl))
        self.weather_manager = WeatherManager(self.input_dir.joinpath(self.ctrl_manager.ctrl_dict['WEATHER_FILE']))
        self.op_manager = OperationManager(self.input_dir.joinpath(self.ctrl_manager.ctrl_dict['OPERATION_FILE']))
        self.crop_manager = CropManager(None)

        # State and action space
        self.crop_obs_size = 14
        self.weather_obs_size = 12
        obs_size = self.crop_obs_size + self.weather_obs_size
        # self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.obs_name = None
        self.action_space = spaces.Discrete(n_actions, )
        self.maxN = maxN
        self.n_actions = n_actions

        # Time info
        self.delta = delta
        self.doy = 1

    def step(self, action):
        # If action is meaningful: rewrite operation file and relaunch simulation. Else: don't simulate, retrieve new state and continue
        # TODO: Setting year this way is only valid for one year simulation
        # TODO: We should make sure there is no fertilizing happening between t and t+delta t. Cannot start with empty operation due to planting and tillage. Cannot remove all fertilization operations otherwise we may loose other nutrients
        year = self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR']
        self.implement_action(action, year=year, doy=self.doy)
        self._call_cycles()
        obs = self.compute_obs(year=year, doy=self.doy)
        self.doy += self.delta
        done = self.doy > 365
        r = self.compute_reward(year, self.doy) if done else 0
        return obs, r, done, {}

    def compute_obs(self, year, doy):
        # TODO: This way we parse every time, we could avoid if the action has not changed
        self.crop_manager.update(self.output_dir.joinpath('CornRM.90.dat'))
        crop_data = self.crop_manager.get_day(year, doy).iloc[0, 4:]
        imm_weather_data = self.weather_manager.immutables.iloc[0, :]
        mutable_weather_data = self.weather_manager.get_day(year, doy).iloc[0, 2:]

        obs = pd.concat([crop_data, imm_weather_data, mutable_weather_data])
        if self.obs_name is None:
            self.obs_name = list(obs.index)
        return obs.to_numpy()

    def compute_reward(self, year, doy):
        return self.crop_manager.get_day(year, doy)['AG BIOMASS']

    def implement_action(self, action, year, doy):
        # TODO: This way if there is a fertilization of other nutrients happening at the same time, we overwrite it
        if action != 0:
            N_mass = action / (self.n_actions - 1) * self.maxN
            year = year - self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR'] + 1  # Convert to operation format
            op = {(year, doy, 'FIXED_FERTILIZATION'): {
              'SOURCE': 'UreaAmmoniumNitrate',
              'MASS': N_mass,
              'FORM': 'Liquid',
              'METHOD': 'Broadcast',
              'LAYER': 1,
              'C_Organic': 0,
              'C_Charcoal': 0,
              'N_Organic': 0,
              'N_Charcoal': 0,
              'N_NH4': 0.75,
              'N_NO3': 0.25,
              'P_Organic': 0,
              'P_CHARCOAL': 0,
              'P_INORGANIC': 0,
              'K': 0,
              'S': 0}}
            self.op_manager.insert_new_operations(op, force=True)
            self.op_manager.write_operation()

    def reset(self):
        # Make sure cycles is executable
        CYCLES_DIR.joinpath('Cycles').chmod(stat.S_IEXEC)

        # # Create operation and control specific to this simulation
        # sim_id = self.get_sim_id()
        # src = self.input_dir.joinpath(self.ctrl_manager.ctrl_dict['OPERATION_FILE'])
        # dest = self.input_dir.joinpath(self.ctrl_manager.ctrl_dict['OPERATION_FILE']).stem + sim_id + '.operation'
        # shutil.copy(src, dest)
        # self.op_manager = OperationManager(dest)


        self._call_cycles()
        self.doy = 1
        return self.compute_obs(year=self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR'], doy=self.doy)

    def render(self, mode="human"):
        pass

    def _call_cycles(self):
        subprocess.run(['./Cycles', '-b', self.ctrl.stem], cwd=CYCLES_DIR)

    @staticmethod
    def get_sim_id():
        return datetime.now().strftime('%Y_%m_%d_%H_%M_%S-') + str(uuid4())


if __name__ == '__main__':
    env = NWinterWheatEnv(CYCLES_DIR.joinpath('input', 'ContinuousCorn.ctrl'))
    env.reset()
    t = time.time()
    for i in range(int(365/7)):
        s, r, done, info = env.step(0)
    print(f'Time elapsed:\t{time.time() - t}')

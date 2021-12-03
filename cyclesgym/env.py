import pathlib
import shutil
from pathlib import Path
import gym
import numpy as np
import pandas as pd
from gym import spaces
import stat
import subprocess
import time
from datetime import datetime
from uuid import uuid4
from cyclesgym.managers import *


from cyclesgym.cycles_config import CYCLES_DIR


class CornEnv(gym.Env):
    def __init__(self, ctrl_base, delta=7, maxN=120, n_actions=7):
        self.input_dir = CYCLES_DIR.joinpath('input')
        self.output_dir = CYCLES_DIR.joinpath('output')
        self.sim_output_dir = None

        self.ctrl_base = Path(ctrl_base).with_suffix('.ctrl')
        self.ctrl = Path(ctrl_base).with_suffix('.ctrl')

        # Check control file exists and initialize managers
        if not self.input_dir.joinpath(self.ctrl).is_file():
            raise ValueError(f'There is no file named {self.ctrl} in {self.input_dir}. A valid control file is necessary'
                             f' to create an environment')
        self.ctrl_base_manager = ControlManager(self.input_dir.joinpath(self.ctrl_base))
        self.ctrl_manager = ControlManager(self.input_dir.joinpath(self.ctrl))
        self.weather_manager = WeatherManager(self.input_dir.joinpath(self.ctrl_manager.ctrl_dict['WEATHER_FILE']))
        self.op_manager = OperationManager(self.input_dir.joinpath(self.ctrl_manager.ctrl_dict['OPERATION_FILE']))
        self.crop_manager = CropManager(None)
        self.season_manager = SeasonManager(None)
        self.sim_id_list = []

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
        year = self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR']
        self._implement_action(action, year=year, doy=self.doy)
        self._call_cycles()
        obs = self.compute_obs(year=year, doy=self.doy)

        self.doy += self.delta
        done = self.doy > 365
        if done:
            self._move_sim_specific_files()
        r = self.compute_reward() if done else 0
        return obs, r, done, {}

    def compute_obs(self, year, doy):
        crop_data = self.crop_manager.get_day(year, doy).iloc[0, 4:]
        imm_weather_data = self.weather_manager.immutables.iloc[0, :]
        mutable_weather_data = self.weather_manager.get_day(year, doy).iloc[0, 2:]

        obs = pd.concat([crop_data, imm_weather_data, mutable_weather_data])
        if self.obs_name is None:
            self.obs_name = list(obs.index)
        return obs.to_numpy()

    def compute_reward(self):
        return self.season_manager.season_df.at[0, 'TOTAL BIOMASS']

    def _implement_action(self, action, year, doy):
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
            self.op_manager.save()

    @staticmethod
    def _udpate_operation(op, N_mass, mode='absolute'):
        """
        Update the NH4 and NO3 values of a fertilization operation.

        Parameters
        ----------
        op: dict
            Dictionary of fertilization operation to update
        N_mass: float
            Total mass of NH4 and NO3 we want to use
        mode: str
            With 'increment' the N_mass is added to the one already present
            in the operation. With 'absolute' we use N_mass regadless of what
            was already there
        Returns
        -------
        new_op: dict
            Dictionary of updated fertilization operation
        """
        assert mode in ['increment', 'absolute']
        new_op = op.copy()
        nutrients = ['C_Organic', 'C_Charcoal', 'N_Organic', 'N_Charcoal', 'N_NH4', 'N_NO3', 'P_Organic',
                     'P_CHARCOAL', 'P_INORGANIC', 'K', 'S']
        new_masses = np.zeros(len(nutrients), dtype=float)

        for i, n in enumerate(nutrients):
            if n == 'N_NH4':
                new_masses[i] = op[n] * op['MASS'] + 0.75 * N_mass if mode == 'increment' else 0.75 * N_mass
            elif n == 'N_NO3':
                new_masses[i] = op[n] * op['MASS'] + 0.25 * N_mass if mode == 'increment' else 0.25 * N_mass
            else:
                new_masses[i] = op[n] * op['MASS']
        total_mass = np.sum(new_masses)
        new_op['MASS'] = total_mass
        new_op.update({nutrient: new_mass / total_mass for (nutrient, new_mass) in zip(nutrients, new_masses)})

        return new_op

    def reset(self):
        # Make sure cycles is executable
        CYCLES_DIR.joinpath('Cycles').chmod(stat.S_IEXEC)

        # Create operation and control specific to this simulation
        current_sim_id = self._create_sim_id()
        self.sim_id_list.append(current_sim_id)
        op_path = self._create_sim_operation_file(current_sim_id)
        self._create_sim_ctrl_file(op_path.name, current_sim_id)
        self.sim_output_dir = self.output_dir.joinpath(self.ctrl.stem)

        self._call_cycles()
        self.doy = 1
        return self.compute_obs(year=self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR'], doy=self.doy)

    def _create_sim_operation_file(self, sim_id):
        """
        Create copy of base operation file specific to this simulation.

        When starting a new simulation, we create a copy of the operation
        file specified in the base_ctrl file. The difference between this
        copy and the original is that all the values of NH4 and NO3 in fixed
        fertilization operations are set to 0. The operation manager
        operates on the copy, not the original.

        Parameters
        ----------
        sim_id: str
            Unique simulation identifier

        Returns
        -------
        dest: pathlib.Path
            Path of the new operation file

        """
        # Copy the operation file and parse it with operation manager
        src = self.input_dir.joinpath(self.ctrl_base_manager.ctrl_dict['OPERATION_FILE'])
        dest = self.input_dir.joinpath(src.stem + sim_id + '.operation')
        shutil.copy(src, dest)
        self.op_manager = OperationManager(dest)

        # Set to zero NH4 and NO3 for existing fertilization operation
        for (year, doy, op_type), op in self.op_manager.op_dict.items():
            if op_type == 'FIXED_FERTILIZATION':
                new_op = self._udpate_operation(op, N_mass=0, mode='absolute')
                self.op_manager.insert_new_operations({(year, doy, op_type): new_op}, force=True)

        # Write operation file to be used in simulation
        self.op_manager.save()
        return dest

    def _create_sim_ctrl_file(self, op_name, sim_id):
        """
        Create copy of base control file specific to this simulation.

        When starting a new simulation, we create a copy of the base control
        file. The difference between this copy and the original is that the
        copy specifies a different operation file (given by op_name,
        which shoul be the output of self._create_sim_operation_file) to run
        the simulation. The control manager operates on the copy, not the
        original.

        Parameters
        ----------
        sim_id: str
            Unique simulation identifier
        op_name: str
            Name of the operation file to be used for the simulation


        Returns
        -------
        dest: pathlib.Path
            Path of the new operation file

        """
        # Store original operation file name in temporary
        tmp = self.ctrl_base_manager.ctrl_dict['OPERATION_FILE']

        # Change to new operation file
        if isinstance(op_name, pathlib.Path):
            op_name = op_name.name
        self.ctrl_base_manager.ctrl_dict['OPERATION_FILE'] = op_name

        # Write new control file
        new_fname = Path(self.ctrl_base.stem + sim_id + '.ctrl')
        dest = self.input_dir.joinpath(new_fname)
        # TODO: Add save method to control manager and use it here
        with open(dest, 'w') as f:
            f.write(self.ctrl_base_manager.to_string())

        # Copy back original operation file name
        self.ctrl_base_manager.ctrl_dict['OPERATION_FILE'] = tmp

        # Change control and its manager
        self.ctrl = new_fname
        self.ctrl_manager = ControlManager(dest)
        return dest

    def _move_sim_specific_files(self):
        if len(self.sim_id_list) > 0:
            fnames = [self.ctrl, Path(self.ctrl_manager.ctrl_dict['OPERATION_FILE'])]
            for fname in fnames:
                self.input_dir.joinpath(fname).rename(self.sim_output_dir.joinpath(fname))

    def render(self, mode="human"):
        pass

    def _call_cycles_raw(self):
        subprocess.run(['./Cycles', '-b', self.ctrl.stem], cwd=CYCLES_DIR)

    def _call_cycles(self):
        self._call_cycles_raw()
        self.crop_manager.update(self.sim_output_dir.joinpath('CornRM.90.dat'))
        self.season_manager.update(self.sim_output_dir.joinpath('season.dat'))

    @staticmethod
    def _create_sim_id():
        return datetime.now().strftime('%Y_%m_%d_%H_%M_%S-') + str(uuid4())

    def close(self):
        try:
            self._move_sim_specific_files()
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    env = CornEnv('ContinuousCorn.ctrl')
    env.reset()
    t = time.time()
    for i in range(365):
        s, r, done, info = env.step(0)
        if done:
            break
    env.close()
    print(f'Time elapsed:\t{time.time() - t}')

import pathlib
import shutil
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import gym
import numpy as np
import pandas as pd
from gym import spaces

from cyclesgym.managers import *
from cyclesgym.paths import CYCLES_PATH


__all__ = ['CornEnv', 'PartialObsCornEnv']


class CornEnv(gym.Env):
    def __init__(self, ctrl_base, delta=7, maxN=120, n_actions=7):
        self.input_dir = CYCLES_PATH.joinpath('input')
        self.output_dir = CYCLES_PATH.joinpath('output')
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
        self.current_op_manager_file = self.input_dir.joinpath(self.ctrl_manager.ctrl_dict['OPERATION_FILE'])
        self.op_manager = OperationManager(self.current_op_manager_file)
        self.crop_manager = CropManager(None)
        self.season_manager = SeasonManager(None)
        self.sim_id_list = []

        # State and action space
        self.crop_obs_size = 14
        self.weather_obs_size = 10
        self.N_to_date = 0
        obs_size = self.crop_obs_size + self.weather_obs_size + 2
        self.observation_space = spaces.Box(low=-10.0,
                                            high=10.0,
                                            shape=(obs_size,),
                                            dtype=np.float32)
        self.obs_name = None
        self.action_space = spaces.Discrete(n_actions, )
        self.maxN = maxN
        self.n_actions = n_actions

        # Time info
        self.delta = delta
        self.doy = 1

        # Rendering
        self.viewer = None
        self.last_action = None

    def _check_new_action(self, action, year, doy):
        """
        Check whether the proposed action is different from the one in the op file.

        Parameters
        ----------
        action: int
            Suggested action
        year: int
            Year of the simulation
        doy: int
            Day of the year
        """
        # Convert year
        year = year - self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR'] + 1

        # Get old operation
        old_op = self.op_manager.op_dict.get((year, doy, 'FIXED_FERTILIZATION'))

        # If there was no fertilization, we need to compare to the "do nothing" action
        if old_op is None:
            return action != 0
        else:
            # Compare if amount of nutrient is the same
            N_mass = action / (self.n_actions - 1) * self.maxN
            return N_mass * 0.75 != old_op['N_NH4'] * old_op['MASS'] or \
                   N_mass * 0.25 != old_op['N_NO3'] * old_op['MASS']

    def step(self, action, debug=False):
        assert self.action_space.contains(action), f'{action} is not a valid action'
        # If action is meaningful: rewrite operation file and relaunch simulation. Else: don't simulate, retrieve new state and continue
        # TODO: Setting year this way is only valid for one year simulation
        year = self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR']
        if self._check_new_action(action, year, self.doy):
            self._implement_action(action, year=year, doy=self.doy)
            self._call_cycles(debug=debug)
        self.N_to_date += self.maxN * action / (self.n_actions - 1)
        obs = self.compute_obs(year=year, doy=self.doy)

        self.doy += self.delta
        done = self.doy > 365
        if done:
            self._move_sim_specific_files()

        r = self.compute_reward(obs, action, done)

        self.last_action = action
        return obs, r, done, {}

    def compute_obs(self, year, doy):
        crop_data = self.crop_manager.get_day(year, doy).iloc[0, 4:]
        imm_weather_data = self.weather_manager.immutables.iloc[0, :]
        mutable_weather_data = self.weather_manager.get_day(year, doy).iloc[0, 2:]

        obs = pd.concat([crop_data, imm_weather_data, mutable_weather_data])
        if self.obs_name is None:
            self.obs_name = list(obs.index) + ['DOY', 'N TO DATE']

        return np.append(obs.to_numpy(dtype=float), [self.doy, self.N_to_date])

    def compute_reward(self, obs, action, done):

        # Penalize fertilization
        N_kg_per_hectare = self.maxN * action / (self.n_actions - 1)

        # Avg anhydrous ammonia cost in 2020 from https://farmdocdaily.illinois.edu/2021/08/2021-fertilizer-price-increases-in-perspective-with-implications-for-2022-costs.html
        N_dollars_per_kg = 496 * 0.001 # $/ton * ton/kg
        N_dollars_per_hectare = N_kg_per_hectare * N_dollars_per_kg
        r = -N_dollars_per_hectare

        # Old reward
        # weight = self.maxN * action / (self.n_actions - 1)
        # r = -25*1e-4 * weight

        # Reward total yield
        if done:
            # Conversion rate for corn from bushel to metric ton from https://grains.org/markets-tools-data/tools/converting-grain-units/
            bushel_per_tonne = 39.3680
            # Avg US price of corn for 2020 from https://quickstats.nass.usda.gov/results/BA8CCB81-A2BB-3C5C-BD23-DBAC365C7832
            dollars_per_bushel = 4.53
            dollars_per_tonne = dollars_per_bushel * bushel_per_tonne

            harvest = self.season_manager.season_df.at[0, 'GRAIN YIELD'] # Metric tonne per hectar
            grain_dollars_per_hectare = harvest * dollars_per_tonne
            r += grain_dollars_per_hectare

        return r

    def _implement_action(self, action, year, doy):
        """
        Write the action to the operation file.

        First, we check for collision. If one occurs, we recompute the operation
        to account for existing nutrient (overwriting NH4 and NO3). If one does
        not occur we just write our action.

        Parameters
        ----------
        action: int
            Suggested action
        year: int
            Year of the simulation
        doy: int
            Day of the year
        """
        if action != 0:
            # Get desired N mass
            N_mass = action / (self.n_actions - 1) * self.maxN

            # Convert year to operation format
            year = year - self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR'] + 1

            # Check for collision
            key = (year, doy, 'FIXED_FERTILIZATION')
            fertilization_op = self.op_manager.op_dict.get(key)
            collision = fertilization_op is not None

            if collision:
                op = {(year, doy, 'FIXED_FERTILIZATION'):
                          self._udpate_operation(fertilization_op, N_mass,
                                                 mode='absolute')}
            else:
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
            self.op_manager.save(self.current_op_manager_file)

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
        if total_mass > 0:
            new_op.update({nutrient: new_mass / total_mass for (nutrient, new_mass) in zip(nutrients, new_masses)})
        else:
            new_op.update({nutrient: 0 for nutrient in nutrients})

        return new_op

    def reset(self, debug=False):
        # Make sure cycles is executable
        CYCLES_PATH.joinpath('Cycles').chmod(stat.S_IEXEC)

        # Create operation and control specific to this simulation
        current_sim_id = self._create_sim_id()
        self.sim_id_list.append(current_sim_id)
        op_path = self._create_sim_operation_file(current_sim_id)
        self._create_sim_ctrl_file(op_path.name, current_sim_id)
        self.sim_output_dir = self.output_dir.joinpath(self.ctrl.stem)

        self._call_cycles(debug=debug)
        self.doy = 1
        self.N_to_date = 0
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
        # TODO: Use update
        self.op_manager = OperationManager(dest)

        # Set to zero NH4 and NO3 for existing fertilization operation
        for (year, doy, op_type), op in self.op_manager.op_dict.items():
            if op_type == 'FIXED_FERTILIZATION':
                new_op = self._udpate_operation(op, N_mass=0, mode='absolute')
                self.op_manager.insert_new_operations({(year, doy, op_type): new_op}, force=True)

        # Write operation file to be used in simulation
        self.current_op_manager_file = dest
        self.op_manager.save(self.current_op_manager_file)
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
            f.write(self.ctrl_base_manager._to_str())

        # Copy back original operation file name
        self.ctrl_base_manager.ctrl_dict['OPERATION_FILE'] = tmp

        # Change control and its manager
        self.ctrl = new_fname
        # TODO: Use update method here
        self.ctrl_manager = ControlManager(dest)
        return dest

    def _move_sim_specific_files(self):
        if len(self.sim_id_list) > 0:
            fnames = [self.ctrl, Path(self.ctrl_manager.ctrl_dict['OPERATION_FILE'])]
            for fname in fnames:
                self.input_dir.joinpath(fname).rename(self.sim_output_dir.joinpath(fname))

    def render(self, mode="human"):
        if self.viewer is None:
            import pyglet
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            fname = Path(__file__).parent.joinpath("assets/green-tea.png")
            self.img_plant = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans_plant = rendering.Transform()
            self.img_plant.add_attr(self.imgtrans_plant)

            fname = Path(__file__).parent.joinpath("assets/chemicals.png")
            self.img_fert = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans_fert = rendering.Transform()
            self.img_fert.add_attr(self.imgtrans_fert)

        #     self.day_label = pyglet.text.Label(
        #         "0000",
        #         font_size=36,
        #         x=20,
        #         y=20 * 2.5 / 40.00,
        #         anchor_x="left",
        #         anchor_y="center",
        #         color=(255, 255, 255, 255),
        #     )
        #
        # self.day_label.text = f"{self.doy:4}"
        # self.day_label.draw()
        # TODO: Add text to credit images

        self.viewer.add_onetime(self.img_plant)
        biomass = self.crop_manager.crop_state.loc[self.doy, 'CUM. BIOMASS']
        self.imgtrans_plant.scale = (-biomass / 10, biomass / 10)

        mass = self.last_action * self.maxN / (self.n_actions - 1) if self.last_action is not None else 0
        if mass > 0:
            self.viewer.add_onetime(self.img_fert)
            self.imgtrans_fert.scale = (-mass / self.maxN, mass / self.maxN)
            self.imgtrans_fert.translation = (- 0.8, 0.8)
        #
        # self.day_label.text = f"{self.doy:4}"
        # self.day_label.draw()

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def _call_cycles_raw(self, debug=False):
        stdout = subprocess.STDOUT if debug else subprocess.DEVNULL
        subprocess.run(['./Cycles', '-b', self.ctrl.stem], cwd=CYCLES_PATH,
                       stdout=stdout)

    def _call_cycles(self, debug=False):
        self._call_cycles_raw(debug=debug)
        self.crop_manager.update_file(self.sim_output_dir.joinpath('CornRM.90.dat'))
        self.season_manager.update_file(self.sim_output_dir.joinpath('season.dat'))

    @staticmethod
    def _create_sim_id():
        return datetime.now().strftime('%Y_%m_%d_%H_%M_%S-') + str(uuid4())

    def close(self):
        try:
            self._move_sim_specific_files()
        except FileNotFoundError:
            pass

        if self.viewer:
            self.viewer.close()
            self.viewer = None


class PartialObsCornEnv(CornEnv):
    def __init__(self, ctrl_base, delta=7, maxN=120, n_actions=7, mask=None):
        super().__init__(ctrl_base, delta, maxN, n_actions)

        if mask is None:
            mask = np.ones(self.observation_space.shape, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)

        if mask.shape != self.observation_space.shape:
            raise ValueError(f'The shape of the observation of the base '
                             f'environment is {self.observation_space.shape},'
                             f'which is different from the mask shape '
                             f'{mask.shape}')
        self.mask = mask
        self.observation_space = spaces.Box(low=self.observation_space.low[mask],
                                            high=self.observation_space.high[mask],
                                            shape=(np.count_nonzero(mask),),
                                            dtype=self.observation_space.dtype)

    def compute_obs(self, year, doy):
        obs = super().compute_obs(year, doy)
        return obs[self.mask]


if __name__ == '__main__':
    import time
    env = CornEnv('ContinuousCorn.ctrl')
    env.reset()
    t = time.time()
    for i in range(365):
        a = np.random.choice(6)
        s, r, done, info = env.step(a)
        if done:
            break
        env.render()
        time.sleep(0.1)
    print(env.obs_name)
    env.close()
    print(f'Time elapsed:\t{time.time() - t}')
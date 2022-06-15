import gym
from copy import copy, deepcopy
import stat
import subprocess
from datetime import date
import os
import numpy as np
from gym import spaces

from cyclesgym.managers import *
from cyclesgym.utils.paths import CYCLES_PATH
from cyclesgym.envs.utils import *


__all__ = ['CyclesEnv', 'PartialObsEnv']


class CyclesEnv(gym.Env):
    def __init__(self,
                 SIMULATION_START_YEAR,
                 SIMULATION_END_YEAR,
                 ROTATION_SIZE,
                 USE_REINITIALIZATION,
                 ADJUSTED_YIELDS,
                 HOURLY_INFILTRATION,
                 AUTOMATIC_NITROGEN,
                 AUTOMATIC_PHOSPHORUS,
                 AUTOMATIC_SULFUR,
                 DAILY_WEATHER_OUT,
                 DAILY_CROP_OUT,
                 DAILY_RESIDUE_OUT,
                 DAILY_WATER_OUT,
                 DAILY_NITROGEN_OUT,
                 DAILY_SOIL_CARBON_OUT,
                 DAILY_SOIL_LYR_CN_OUT,
                 ANNUAL_SOIL_OUT,
                 ANNUAL_PROFILE_OUT,
                 ANNUAL_NFLUX_OUT,
                 CROP_FILE,
                 OPERATION_FILE,
                 SOIL_FILE,
                 WEATHER_FILE,
                 REINIT_FILE,
                 delta,
                 *args,
                 **kwargs):

        d = {
            'SIMULATION_START_YEAR': SIMULATION_START_YEAR,
            'SIMULATION_END_YEAR': SIMULATION_END_YEAR,
            'ROTATION_SIZE': ROTATION_SIZE,
            'USE_REINITIALIZATION': USE_REINITIALIZATION,
            'ADJUSTED_YIELDS': ADJUSTED_YIELDS,
            'HOURLY_INFILTRATION': HOURLY_INFILTRATION,
            'AUTOMATIC_NITROGEN': AUTOMATIC_NITROGEN,
            'AUTOMATIC_PHOSPHORUS': AUTOMATIC_PHOSPHORUS,
            'AUTOMATIC_SULFUR': AUTOMATIC_SULFUR,
            'DAILY_WEATHER_OUT': DAILY_WEATHER_OUT,
            'DAILY_CROP_OUT': DAILY_CROP_OUT,
            'DAILY_RESIDUE_OUT': DAILY_RESIDUE_OUT,
            'DAILY_WATER_OUT': DAILY_WATER_OUT,
            'DAILY_NITROGEN_OUT': DAILY_NITROGEN_OUT,
            'DAILY_SOIL_CARBON_OUT': DAILY_SOIL_CARBON_OUT,
            'DAILY_SOIL_LYR_CN_OUT': DAILY_SOIL_LYR_CN_OUT,
            'ANNUAL_SOIL_OUT': ANNUAL_SOIL_OUT,
            'ANNUAL_PROFILE_OUT': ANNUAL_PROFILE_OUT,
            'ANNUAL_NFLUX_OUT': ANNUAL_NFLUX_OUT,
            'CROP_FILE': CROP_FILE,
            'OPERATION_FILE': OPERATION_FILE,
            'SOIL_FILE': SOIL_FILE,
            'WEATHER_FILE': WEATHER_FILE,
            'REINIT_FILE': REINIT_FILE,
        }

        # Base control containing all the defaults
        self.ctrl_base_manager = ControlManager.from_dict(d)

        # Simulation specific control used to determine sim-specific weather,
        # operation, and similar
        self.ctrl_manager = copy(self.ctrl_base_manager)
        self.ctrl_file = None

        # Base operation with all the defaults (e.g. it may specify planting in
        # simulation where the agent can only take fertilization actions)
        self.op_base_file = CYCLES_PATH.joinpath(
            'input', self.ctrl_base_manager.ctrl_dict['OPERATION_FILE'])
        self.op_base_manager = OperationManager(self.op_base_file)

        # Sim specific operation
        self.op_manager = None
        self.op_file = None

        # Other simulation related files
        self.crop_input_file = None
        self.soil_input_file = None
        self.weather_input_file = None

        # Sim specific I/O directories and sim identifier
        self.simID = None
        self.input_dir = None
        self.output_dir = None

        # Classes to compute states, rewards, implement actions
        self.observer = None
        self.implementer = None
        self.rewarder = None

        # Obs and action space
        self.observation_space = None
        self.action_space = None

        # Date
        self.date = None
        self.delta = delta

        # Input files managers
        self.input_managers = []
        self.input_files = []

        # Output files managers
        self.output_managers = []
        self.output_files = []

    def _post_init_setup(self):
        self.weather_manager = None
        self.crop_output_file = None
        self.crop_output_manager = None
        self.season_file = None
        self.season_manager = None

    def _create_io_dirs(self):
        """
        Create temporary directory for input and output files.
        """
        self.simID = create_sim_id()
        self.input_dir = MyTemporaryDirectory(
            path=CYCLES_PATH.joinpath('input', self.simID))
        self.output_dir = MyTemporaryDirectory(
            path=CYCLES_PATH.joinpath('output', self.simID))

    def _get_output_dir(self):
        """
        Get path where output files are stored.

        Cycles automatically creates a directory with the same name as the
        control file in the output directory to store the output. This function
        returns a path pointing to such folder.
        """
        return self.output_dir.name.joinpath('control')

    def _create_operation_file(self):
        """Create operation file by copying the base one."""
        self.op_manager = deepcopy(self.op_base_manager)
        self.op_file = Path(self.input_dir.name).joinpath(
            'operation.operation')
        self.op_manager.save(self.op_file)

    def _create_crop_input_file(self):
        """Creat crop file by simlinking the one indicated in the base ctrl."""
        crop_file = self.ctrl_base_manager.ctrl_dict['CROP_FILE']
        src = CYCLES_PATH.joinpath('input', crop_file)
        if not src.exists():
            raise ValueError(f'There is no {crop_file}  in '
                             f'{CYCLES_PATH.joinpath("input")}')
        dest = self.input_dir.name.joinpath('crop.crop')
        self.crop_input_file = dest
        os.symlink(src, dest)

    def _create_soil_input_file(self):
        """Creat soil file by simlinking the one indicated in the base ctrl."""
        soil_file = self.ctrl_base_manager.ctrl_dict['SOIL_FILE']
        src = CYCLES_PATH.joinpath('input', soil_file)
        if not src.exists():
            raise ValueError(f'There is no {soil_file}  in '
                             f'{CYCLES_PATH.joinpath("input")}')
        dest = self.input_dir.name.joinpath('soil.soil')
        self.soil_input_file = dest
        os.symlink(src, dest)

    def _create_weather_input_file(self):
        """Creat weather file by simlinking the one indicated in the base ctrl."""
        weather_file = self.ctrl_base_manager.ctrl_dict['WEATHER_FILE']
        src = CYCLES_PATH.joinpath('input', weather_file)
        if not src.exists():
            raise ValueError(f'There is no {weather_file}  in '
                             f'{CYCLES_PATH.joinpath("input")}')
        dest = self.input_dir.name.joinpath('weather.weather')
        self.weather_input_file = dest
        os.symlink(src, dest)

    def _create_control_file(self):
        """Create control file pointing to right input files."""
        # Copy base control
        self.ctrl_manager = copy(self.ctrl_base_manager)

        # Point to right input files
        self.ctrl_manager.ctrl_dict['OPERATION_FILE'] = \
            str(Path(*self.op_file.parts[-2:]))
        self.ctrl_manager.ctrl_dict['CROP_FILE'] = \
            str(Path(*self.crop_input_file.parts[-2:]))
        self.ctrl_manager.ctrl_dict['SOIL_FILE'] = \
            str(Path(*self.soil_input_file.parts[-2:]))
        self.ctrl_manager.ctrl_dict['WEATHER_FILE'] = \
            str(Path(*self.weather_input_file.parts[-2:]))

        # Write control file in input directory
        self.ctrl_file = Path(self.input_dir.name).joinpath('control.ctrl')
        self.ctrl_manager.save(self.ctrl_file)

    def _init_input_managers(self):
        """
        Initialize all the input file managers and their file paths.
        """
        raise NotImplementedError

    def _udpate_input_managers(self):
        """
        Update the input file managers based on the corresponding file paths.
        """
        for manager, file in zip(self.input_managers, self.input_files):
            manager.update_file(file)

    def _init_output_managers(self):
        """
        Initialize all the output file managers and their file paths.
        """
        raise NotImplementedError

    def _update_output_managers(self):
        """
        Update the output file managers based on the corresponding file paths.
        """
        # TODO: give a warning if self.output_managers, self.output_files not of same lenght
        for manager, file in zip(self.output_managers, self.output_files):
            manager.update_file(file)

    def _init_observer(self, *args, **kwargs):
        """
        Initialize state observer.
        """
        raise NotImplementedError

    def _init_rewarder(self):
        raise NotImplementedError

    def _init_implementer(self):
        raise NotImplementedError

    def _common_reset(self):
        """
        Reset steps that must be performed regardless of observer and so on.

        In particular, we make cycles executable, start tracking the date,
        generate all the simulation specific files in the dedicated directory,
        and run cycles (which saves all the outputs in the dedicated directory).
        """
        # Make sure cycles is executable
        CYCLES_PATH.joinpath('Cycles').chmod(stat.S_IEXEC)

        # Init date
        self.date = date(
            year=self.ctrl_base_manager.ctrl_dict['SIMULATION_START_YEAR'],
            month=1, day=1)

        # Prepare cycles files
        self._create_io_dirs()
        self._create_crop_input_file()
        self._create_weather_input_file()
        self._create_soil_input_file()
        self._create_operation_file()
        self._create_control_file()
        self._call_cycles_raw(debug=True)

        # Initialize managers
        self._init_input_managers()
        self._init_output_managers()

    def reset(self):
        """
        Reset an episode.

        First, it calls _common_reset to generate all the files and run cycles.
        Subsequently, it initializes all the managers, observers, rewarders,
        implementers, and returns the inital state. It is separated from the
        _common_reset as different observers may require different inputs.
        """
        raise NotImplementedError

    def step(self, action):
        pass

    def render(self, mode="human"):
        pass

    def _call_cycles_raw(self, debug=False, reinit=False, doy=None):
        input_file = str(Path(*self.ctrl_file.parts[-2:])).replace('.ctrl', '')

        # Redirect cycles output unless we are debugging

        strings = ['-b']
        if debug:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        if reinit and doy:
            strings.append('-l')
            strings.append(str(doy))

        strings.append(input_file)

        subprocess.run(['./Cycles', *strings], cwd=CYCLES_PATH, stdout=stdout)

    def _call_cycles(self, debug=False, reinit=False, doy=None):
        self._call_cycles_raw(debug=debug, reinit=reinit, doy=doy)
        self._update_output_managers()


class PartialObsEnv(gym.ObservationWrapper):
    def __init__(self, env, mask=None):
        super().__init__(env)

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

        self.observation_space = spaces.Box(
            low=self.env.observation_space.low[mask],
            high=self.env.observation_space.high[mask],
            shape=(np.count_nonzero(mask),),
            dtype=self.env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        if len(self.env.observer.obs_names) > np.count_nonzero(self.mask):
            obs_names = np.asarray(self.env.observer.obs_names)[self.mask]
            self.env.observer.obs_names = list(obs_names)
        return obs

    def observation(self, obs):
        return obs[self.mask]



if __name__ == '__main__':
    env = CyclesEnv(
        SIMULATION_START_YEAR=1980,
        SIMULATION_END_YEAR = 1980,
        ROTATION_SIZE = 1,
        USE_REINITIALIZATION = 0,
        ADJUSTED_YIELDS = 0,
        HOURLY_INFILTRATION = 1,
        AUTOMATIC_NITROGEN = 0,
        AUTOMATIC_PHOSPHORUS = 0,
        AUTOMATIC_SULFUR = 0,
        DAILY_WEATHER_OUT = 1,
        DAILY_CROP_OUT = 1,
        DAILY_RESIDUE_OUT = 1,
        DAILY_WATER_OUT = 1,
        DAILY_NITROGEN_OUT = 1,
        DAILY_SOIL_CARBON_OUT = 1,
        DAILY_SOIL_LYR_CN_OUT = 1,
        ANNUAL_SOIL_OUT = 1,
        ANNUAL_PROFILE_OUT = 1,
        ANNUAL_NFLUX_OUT = 1,
        CROP_FILE = 'GenericCrops.crop',
        OPERATION_FILE = 'ContinuousCorn.operation',
        SOIL_FILE = 'GenericHagerstown.soil',
        WEATHER_FILE = 'RockSprings.weather',
        REINIT_FILE='N / A',
        delta=7)
    env._common_reset()


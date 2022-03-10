import gym
from copy import copy, deepcopy
import stat
import subprocess

from cyclesgym.managers import *
from cyclesgym.paths import CYCLES_PATH, PROJECT_PATH
from cyclesgym.envs.utils import *


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

        self.ctrl_base_manager = ControlManager.from_dict(d)
        self.ctrl_manager = copy(self.ctrl_base_manager)
        self.ctrl_file = None
        self.op_base_manager = None
        self.op_base_file = CYCLES_PATH.joinpath(
            'input', self.ctrl_base_manager.ctrl_dict['OPERATION_FILE'])
        self.op_manager = None
        self.op_file = None

        self.input_dir = None
        self.output_dir = None
        self.simID = None

        self.observation_space = None
        self.action_space = None

    def _init_managers(self):
        raise NotImplementedError

    def reset(self):
        # Make sure cycles is executable
        CYCLES_PATH.joinpath('Cycles').chmod(stat.S_IEXEC)

        # Create temporary directory for input and output files of this episode
        self.simID = create_sim_id()
        self.input_dir = MyTemporaryDirectory(
            path=CYCLES_PATH.joinpath('input', self.simID))
        self.output_dir = MyTemporaryDirectory(
            path=CYCLES_PATH.joinpath('output', self.simID))

        # Copy the manager containing the base operations and save it in the temporary input directory
        self.op_manager = deepcopy(self.op_base_manager)
        self.op_file = Path(self.input_dir.name).joinpath('operation.operation')
        # self.op_manager.save(self.op_manager_file)
        # !!!!!!!!!!This is a temporary operation file, remove this line
        with open(self.op_file, 'w') as fp:
            fp.write('\n')

        self.ctrl_manager = copy(self.ctrl_base_manager)
        self.ctrl_manager.ctrl_dict['OPERATION_FILE'] = \
            str(Path(*self.op_file.parts[-2:]))
        self.ctrl_file = Path(self.input_dir.name).joinpath('control.ctrl')
        self.ctrl_manager.save(self.ctrl_file)
        env._call_cycles(debug=True)

    def step(self, action):
        pass

    def render(self, mode="human"):
        pass

    def _call_cycles_raw(self, debug=False):
        input_file = str(Path(*self.ctrl_file.parts[-2:])).replace('.ctrl', '')

        # Redirect cycles output unless we are debugging
        if debug:
            subprocess.run(['./Cycles', '-b', input_file], cwd=CYCLES_PATH,
                           stdout=None)
        else:
            subprocess.run(['./Cycles', '-b', input_file], cwd=CYCLES_PATH,
                           stdout=subprocess.DEVNULL)

    def _call_cycles(self, debug=False):
        self._call_cycles_raw(debug=debug)


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
    env.reset()


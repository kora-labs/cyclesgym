from cyclesgym.envs.common import CyclesEnv
from cyclesgym.envs.observers import SoilNObserver, CropRotationTrailingWindowObserver
from cyclesgym.envs.rewarders import CropRewarder, compound_rewarder
from cyclesgym.envs.implementers import RotationPlanter, RotationPlanterFixedPlanting
from cyclesgym.managers import WeatherManager, CropManager, SeasonManager, OperationManager, SoilNManager
from cyclesgym.utils.paths import CYCLES_PATH
from cyclesgym.envs.weather_generator import WeatherShuffler, FixedWeatherGenerator

from gym import spaces
from typing import Tuple
import numpy as np
from datetime import date
import os
from pathlib import Path


class CropPlanning(CyclesEnv):
    def __init__(self,
                 start_year,
                 end_year,
                 rotation_crops,
                 soil_file='GenericHagerstown.soil',
                 weather_generator_class=FixedWeatherGenerator,
                 weather_generator_kwargs={
                     'base_weather_file': CYCLES_PATH.joinpath('input', 'RockSprings.weather')}
                 ):

        super().__init__(SIMULATION_START_YEAR=start_year,
                         SIMULATION_END_YEAR=end_year,
                         ROTATION_SIZE=end_year - start_year + 1,
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
                         DAILY_NITROGEN_OUT=1,
                         DAILY_SOIL_CARBON_OUT=0,
                         DAILY_SOIL_LYR_CN_OUT=0,
                         ANNUAL_SOIL_OUT=0,
                         ANNUAL_PROFILE_OUT=0,
                         ANNUAL_NFLUX_OUT=0,
                         CROP_FILE='GenericCrops.crop',
                         OPERATION_FILE='CornSilageSoyWheat.operation',
                         #TODO: right now the operation file is totally ignored
                         SOIL_FILE=soil_file,
                         WEATHER_GENERATOR_CLASS=weather_generator_class,
                         WEATHER_GENERATOR_KWARGS=weather_generator_kwargs,
                         REINIT_FILE='N / A',
                         delta=365)
        self.rotation_crops = rotation_crops
        self.reset()
        self._generate_observation_space()
        self._generate_action_space(len(rotation_crops))

    def _generate_action_space(self, n_actions):
        self.action_space = spaces.MultiDiscrete([n_actions, 14, 10, 10])
        self.n_actions = n_actions

    def _generate_observation_space(self):
        self.observation_space = spaces.Box(
            low=np.array(self.observer.lower_bound, dtype=np.float32),
            high=np.array(self.observer.lower_bound, dtype=np.float32),
            shape=self.observer.lower_bound.shape,
            dtype=np.float32)

    def _init_input_managers(self):
        #TODO: in common with Corn environment. Should be implemented in and abstract parent class
        self.weather_manager = WeatherManager(self.weather_input_file)
        self.input_managers = [self.weather_manager]
        self.input_files = [self.weather_input_file]

    def _init_implementer(self, *args, **kwargs):
        self.implementer = RotationPlanter(
            operation_manager=self.op_manager,
            operation_fname=self.op_file,
            rotation_crops=self.rotation_crops,
            start_year=self.ctrl_base_manager.ctrl_dict['SIMULATION_START_YEAR']
        )

    def _init_output_managers(self):
        self.crop_output_file = [self._get_output_dir().joinpath(crop + '.dat') for crop in self.rotation_crops]
        self.season_file = self._get_output_dir().joinpath('season.dat')
        self.soil_n_file = self._get_output_dir().joinpath('N.dat')

        for file in self.crop_output_file:
            if not os.path.exists(file):
                with open(file, 'w'): pass

        self.crop_output_manager = [CropManager(file) for file in self.crop_output_file]
        self.season_manager = SeasonManager(self.season_file)
        self.soil_n_manager = SoilNManager(self.soil_n_file)

        self.output_managers = [*self.crop_output_manager,
                                self.season_manager,
                                self.soil_n_manager]
        self.output_files = [*self.crop_output_file,
                             self.season_file,
                             self.soil_n_file]

    def _init_observer(self, *args, **kwargs):
        self.observer = SoilNObserver(soil_n_manager=self.soil_n_manager,
                                      end_year=self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR'])

    def _init_rewarder(self, *args, **kwargs):
        self.rewarder = compound_rewarder([CropRewarder(self.season_manager, name)
                                           for name in self.rotation_crops])

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        rerun_cycles = self.implementer.implement_action(self.date, *action)

        if rerun_cycles:
            self._call_cycles(debug=False)

        # Advance time
        self.date = date(self.date.year + 1, self.date.month, self.date.day)

        # Compute reward
        r = self.rewarder.compute_reward(date=self.date, delta=self.delta)

        # Compute state
        obs = self.observer.compute_obs(self.date, action=action)

        # Compute
        done = self.date.year > self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR']

        return obs, r, done, {}

    def _create_operation_file(self):
        # deleting all the content of the operation file found.
        self.op_file = Path(self.input_dir.name).joinpath(
            'operation.operation')
        open(self.op_file, 'w').close()
        self.op_manager = OperationManager(self.op_file)
        self.op_base_manager = OperationManager(self.op_file)

    def reset(self) -> np.ndarray:
        # Set up dirs and files and run first simulation
        self._common_reset()

        # Init objects to compute obs, rewards, and implement actions
        self._init_observer()
        self._init_rewarder()
        self._init_implementer()

        # Set to zero all pre-existing fertilization for N
        self.implementer.reset()
        obs = self.observer.compute_obs(self.date, action=[-1, 0])
        self.observer.reset()
        return obs


class CropPlanningFixedPlanting(CropPlanning):

    def _generate_action_space(self, n_actions):
        self.action_space = spaces.MultiDiscrete([n_actions, 14])
        self.n_actions = n_actions

    def _init_implementer(self, *args, **kwargs):
        self.implementer = RotationPlanterFixedPlanting(
            operation_manager=self.op_manager,
            operation_fname=self.op_file,
            rotation_crops=self.rotation_crops,
            start_year=self.ctrl_base_manager.ctrl_dict['SIMULATION_START_YEAR']
        )


class CropPlanningFixedPlantingRotationObserver(CropPlanningFixedPlanting):

    def _init_observer(self, *args, **kwargs):
        self.observer = CropRotationTrailingWindowObserver(
            end_year=self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR'])


if __name__ == '__main__':
    # Example for crop plannign with random weather

    # Env kwargs
    start_year = 1980
    end_year = 1990
    rotation_crops = ['CornSilageRM.90', 'SoybeanMG.3']
    soil_file = 'GenericHagerstown.soil'
    weather_file = 'RockSprings.weather'

    # Weather generator
    sampling_start_year = 1980
    sampling_end_year = 2015
    n_weather_samples = 10
    target_year_range = np.arange(start_year, end_year + 1)
    weather_generator_kwargs = dict(n_weather_samples=n_weather_samples,
                                    sampling_start_year=sampling_start_year,
                                    sampling_end_year=sampling_end_year,
                                    base_weather_file=CYCLES_PATH.joinpath('input', weather_file),
                                    target_year_range=target_year_range)

    env = CropPlanningFixedPlanting(start_year=start_year,
                                    end_year=end_year,
                                    rotation_crops=rotation_crops,
                                    soil_file=soil_file,
                                    weather_generator_class=WeatherShuffler,
                                    weather_generator_kwargs=weather_generator_kwargs)

    # Compare with loop
    n_trials = 5

    rewards = np.zeros(n_trials)

    for i in range(n_trials):
        env.reset()  # Fix weather file
        year = 0
        while True:
            a = (0, 0) if np.mod(year, 2) == 0 else (1, 0)
            s, r, done, info = env.step(a)
            rewards[0, i] += r
            year += 1
            if done:
                break

    print(rewards)



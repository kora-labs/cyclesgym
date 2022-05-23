from cyclesgym.envs.common import CyclesEnv
from cyclesgym.envs.observers import WheatherObserver, compound_observer, CropObserver
from cyclesgym.envs.rewarders import CropRewarder, compound_rewarder
from cyclesgym.envs.implementers import RotationPlanter
from cyclesgym.managers import WeatherManager, CropManager, SeasonManager, OperationManager
from gym import spaces
from typing import Tuple
import numpy as np
from datetime import date
import os
from pathlib import Path


class CropPlanning(CyclesEnv):
    def __init__(self, start_year, end_year, rotation_crops):
        super().__init__(SIMULATION_START_YEAR=start_year,
                         SIMULATION_END_YEAR=end_year,
                         ROTATION_SIZE=end_year-start_year,
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
                         ANNUAL_SOIL_OUT=1,
                         ANNUAL_PROFILE_OUT=0,
                         ANNUAL_NFLUX_OUT=0,
                         CROP_FILE='GenericCrops.crop',
                         OPERATION_FILE='CornSilageSoyWheat.operation',
                         #TODO: right now the operation file is totally ignored
                         SOIL_FILE='GenericHagerstown.soil',
                         WEATHER_FILE='RockSprings.weather',
                         REINIT_FILE='N / A',
                         delta=365)
        self._post_init_setup()
        self._generate_action_space(len(rotation_crops))
        self.rotation_crops = rotation_crops

    def _generate_action_space(self, n_actions):
        self.action_space = spaces.Tuple((spaces.Discrete(n_actions,),
                                          spaces.Box(low=np.float32(0), high=np.float32(1.0), shape=[1]),
                                          spaces.Box(low=np.float32(0), high=np.float32(1.0), shape=[1]),
                                          spaces.Box(low=np.float32(0), high=np.float32(1.0), shape=[1])))
        self.n_actions = n_actions

    def _generate_observation_space(self):
        self.observation_space = spaces.Box(
            low=np.array(WeatherCropDoyNObserver.lower_bound,dtype=np.float32),
            high=np.array(WeatherCropDoyNObserver.lower_bound,dtype=np.float32),
            shape=WeatherCropObserver.lower_bound.shape,
            dtype=np.float32)
        #TODO: write observation of soil data

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

        for file in self.crop_output_file:
            if not os.path.exists(file):
                with open(file, 'w'): pass

        self.crop_output_manager = [CropManager(file) for file in self.crop_output_file]
        self.season_manager = SeasonManager(self.season_file)

        self.output_managers = [*self.crop_output_manager,
                                self.season_manager]
        self.output_files = [*self.crop_output_file,
                             self.season_file]

    def _init_observer(self, *args, **kwargs):
        #TODO: add obersavtion of soil
        obs_list = [WheatherObserver(weather_manager=self.weather_manager,
                                    end_year=self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR'])]
        obs_list = obs_list + [CropObserver(crop_manager=crop_man,
                                            end_year=self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR'])
                               for crop_man in self.crop_output_manager]
        self.observer = compound_observer(obs_list)

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
        obs = self.observer.compute_obs(self.date)

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
        return self.observer.compute_obs(self.date)
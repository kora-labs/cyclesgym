from cyclesgym.envs.common import CyclesEnv
from cyclesgym.envs.observers import WheatherObserver, compound_observer, CropObserver
from cyclesgym.envs.rewarders import CornNProfitabilityRewarder
from cyclesgym.envs.implementers import RotationPlanter
from cyclesgym.managers import WeatherManager, CropManager, SeasonManager
from gym import spaces
from typing import Tuple
import numpy as np
from datetime import date


class CropPlannig(CyclesEnv):
    def __init__(self, rotation_crops):
        super().__init__(SIMULATION_START_YEAR=1980,
                         SIMULATION_END_YEAR=1990,
                         ROTATION_SIZE=10,
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
                         SOIL_FILE='GenericHagerstown.soil',
                         WEATHER_FILE='RockSprings.weather',
                         REINIT_FILE='N / A',
                         delta=365)
        self._post_init_setup()
        self._generate_action_space(len(rotation_crops))
        self.rotation_crops = rotation_crops

    def _generate_action_space(self, n_actions):
        self.action_space = spaces.Tuple((spaces.Discrete(n_actions,),
                                          spaces.Box(low=0, high=1.0, shape=[1]),
                                          spaces.Box(low=0, high=1.0, shape=[1]),
                                          spaces.Box(low=0, high=1.0, shape=[1])))
        self.n_actions = n_actions

    def _generate_observation_space(self):
        self.observation_space = spaces.Box(
            low=WeatherCropObserver.lower_bound,
            high=WeatherCropObserver.upper_bound,
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
        # TODO: add rewarder for every crop
        self.rewarder = CornNProfitabilityRewarder(self.season_manager)

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        rerun_cycles = self.implementer.implement_action(self.date, *action)

        if rerun_cycles:
            self._call_cycles(debug=False)

        # Advance time
        self.date = date(self.date.year + 1, self.date.month, self.date.day)

        # Compute reward
        r = self.rewarder.compute_reward(Nkg_per_heactare=0, date=self.date, delta=self.delta)

        # Compute state
        obs = self.observer.compute_obs(self.date)

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
        return self.observer.compute_obs(self.date)
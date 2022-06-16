from datetime import timedelta
from cyclesgym.envs.common import CyclesEnv
from cyclesgym.envs.observers import compound_observer, CropObserver, \
    WeatherObserver, NToDateObserver
from cyclesgym.envs.rewarders import compound_rewarder, CropRewarder, \
    NProfitabilityRewarder
from cyclesgym.utils.paths import CYCLES_PATH
from cyclesgym.envs.weather_generator import WeatherShuffler, FixedWeatherGenerator
import os
from cyclesgym.envs.constrainers import FertilizationEventConstrainer, \
    TotalNitrogenConstrainer, LeachingConstrainer, compound_constrainer

from cyclesgym.envs.implementers import *
import pathlib
import shutil
from typing import Tuple

import numpy as np
from gym import spaces


from cyclesgym.managers import *

__all__ = ['Corn']


class Corn(CyclesEnv):
    def __init__(self, delta,
                 n_actions,
                 maxN,
                 operation_file='ContinuousCorn.operation',
                 soil_file='GenericHagerstown.soil',
                 weather_generator_class=FixedWeatherGenerator,
                 weather_generator_kwargs={
                     'base_weather_file': CYCLES_PATH.joinpath('input', 'RockSprings.weather')},
                 start_year=1980,
                 end_year=1980,
                 use_reinit=True
                 ):
        self.rotation_size = end_year - start_year + 1
        self.use_reinit = use_reinit
        super().__init__(SIMULATION_START_YEAR=start_year,
                         SIMULATION_END_YEAR=end_year,
                         ROTATION_SIZE=self.rotation_size,
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
                         OPERATION_FILE=operation_file,
                         SOIL_FILE=soil_file,
                         WEATHER_GENERATOR_CLASS=weather_generator_class,
                         WEATHER_GENERATOR_KWARGS=weather_generator_kwargs,
                         REINIT_FILE='N / A',
                         delta=delta)
        self._post_init_setup()
        self._init_observer()
        self._generate_observation_space()
        self._generate_action_space(n_actions, maxN)

        # TODO: Move to CyclesEnv
        self.constrainer = None

    def _post_init_setup(self):
        super()._post_init_setup()
        self.soil_n_file = None
        self.soil_n_manager = None

    def _generate_action_space(self, n_actions, maxN):
        self.action_space = spaces.Discrete(n_actions, )
        self.maxN = maxN
        self.n_actions = n_actions

    def _generate_observation_space(self):
        self.observation_space = spaces.Box(
            low=np.array(self.observer.lower_bound, dtype=np.float32),
            high=np.array(self.observer.upper_bound, dtype=np.float32),
            shape=self.observer.lower_bound.shape,
            dtype=np.float32)

    def _init_input_managers(self):
        self.weather_manager = WeatherManager(self.weather_input_file)
        self.input_managers = [self.weather_manager]
        self.input_files = [self.weather_input_file]

    def _init_output_managers(self):
        # Files
        self.crop_output_file = self._get_output_dir().joinpath('CornRM.90.dat')
        self.season_file = self._get_output_dir().joinpath('season.dat')
        self.soil_n_file = self._get_output_dir().joinpath('N.dat')

        # Managers
        self.crop_output_manager = CropManager(self.crop_output_file)
        self.season_manager = SeasonManager(self.season_file)
        self.soil_n_manager = SoilNManager(self.soil_n_file)

        self.output_managers = [self.crop_output_manager,
                                self.season_manager,
                                self.soil_n_manager]
        self.output_files = [self.crop_output_file,
                             self.season_file,
                             self.soil_n_file]

    def _init_observer(self, *args, **kwargs):
        end_year = self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR']
        self.observer = compound_observer([WeatherObserver(weather_manager=self.weather_manager, end_year=end_year),
                                           CropObserver(crop_manager=self.crop_output_manager, end_year=end_year),
                                           NToDateObserver(end_year=end_year)
                                           ])

    def _init_rewarder(self, *args, **kwargs):
        self.rewarder = compound_rewarder([CropRewarder(self.season_manager, 'CornRM.90'),
                                           NProfitabilityRewarder()])

    def _init_implementer(self, *args, **kwargs):
        self.implementer = FixedRateNFertilizer(
            operation_manager=self.op_manager,
            operation_fname=self.op_file,
            rate=0.75,
            start_year=self.ctrl_base_manager.ctrl_dict['SIMULATION_START_YEAR']
        )

    def _init_constrainer(self):
        # Initialize constrainer
        end_year = self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR']
        self.constrainer = compound_constrainer(
            [TotalNitrogenConstrainer(),
             FertilizationEventConstrainer(),
             LeachingConstrainer(soil_n_manager=self.soil_n_manager,
                                 end_year=end_year)])

    def _action2mass(self, action: int) -> float:
        return self.maxN * action / (self.n_actions - 1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert self.action_space.contains(action), f'{action} is not contained in the action space'
        action = self._action2mass(action)
        rerun_cycles = self.implementer.implement_action(
            date=self.date, mass=action)

        doy = None
        reinit = False
        if self.use_reinit:
            reinit = self._check_is_mid_year()
            if reinit:
                doy = 365

        if rerun_cycles or reinit:
            self._call_cycles(debug=False, reinit=reinit, doy=doy)

        # Advance time
        self.date += timedelta(days=self.delta)

        if reinit:
            self._update_control_file()
            self._update_reinit_file()
            self._update_operation_file()
            self.implementer.start_year = self.reinit_year + 1

        # TODO: output managers are updated in cycles call below. Should we remove this?
        self._update_output_managers()

        done = self.date.year > self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR']

        if reinit and not done:
            self._call_cycles(debug=False)

        # Compute reward
        r = self.rewarder.compute_reward(date=self.date, delta=self.delta,
                                         action=action)
        # Compute constraints values
        info = {}
        constraints = self.constrainer.compute_constraint(date=self.date,
                                                          action=action)
        info.update(constraints)

        # Compute
        obs = self.observer.compute_obs(self.date, N=action)

        return obs, r, done, info

    def reset(self) -> np.ndarray:
        # Set up dirs and files and run first simulation
        self._common_reset()

        # Init objects to compute obs, rewards, and implement actions
        self._init_observer()
        self._init_rewarder()
        self._init_implementer()
        self._init_constrainer()

        # Set to zero all pre-existing fertilization for N
        rerun_cycles = self.implementer.reset()
        if rerun_cycles:
            self._call_cycles(debug=False, reinit=False, doy=None)
        return self.observer.compute_obs(self.date, N=0)

    def _create_operation_file(self):
        """Create operation file by copying the base one."""
        super(Corn, self)._create_operation_file()
        operations = [key for key in self.op_manager.op_dict.keys() if key[2] != 'FIXED_FERTILIZATION']
        for i in range(self.rotation_size - 1):
            for op in operations:
                copied_op = (i + 2,) + op[1:]
                if not any(key[0] == copied_op[0] and key[2] == copied_op[2] for key in operations):
                    self.op_manager.op_dict[copied_op] = self.op_manager.op_dict[op]
                    print(f'Copying operation {copied_op} into the operation file, as no operation'
                          f' of the same kind is available for that year.')

        self.op_manager.save(self.op_file)

    def _create_control_file(self):
        super(Corn, self)._create_control_file()
        if self.use_reinit:
            self.reinit_year = int((self.ctrl_base_manager.ctrl_dict['SIMULATION_START_YEAR']
                                    + self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR'])/2)
            self.ctrl_manager.ctrl_dict['SIMULATION_END_YEAR'] = self.reinit_year
            self.ctrl_manager.save(self.ctrl_file)

    def _update_reinit_file(self):
        new_reinit = self.input_dir.name.joinpath('reinit.dat')
        shutil.copy(self._get_output_dir().joinpath('reinit.dat'), new_reinit)

        lines = []
        with open(new_reinit, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line_splitted = line.split()
                if len(line_splitted) > 0:
                    if line_splitted[1].isnumeric():
                        if int(line_splitted[1]) == self.reinit_year:
                            indx = i
                            line = line_splitted
                            line[1] = str(int(line[1]) + 1)
                            line[3] = '1'
                            line = line[0] + '    ' + line[1] + '    ' + line[2] + '     ' + line[3] + '\n'
                lines.append(line)

        with open(new_reinit, 'w') as f:
            for i, line in enumerate(lines):
                if i >= indx:
                    f.write(line)

    def _update_control_file(self):
        self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR'] = self.reinit_year + 1
        self.ctrl_manager.ctrl_dict['SIMULATION_END_YEAR'] = self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR']
        self.ctrl_manager.ctrl_dict['USE_REINITIALIZATION'] = 1
        self.ctrl_manager.ctrl_dict['REINIT_FILE'] = pathlib.Path(self.input_dir.name.stem).joinpath('reinit.dat')
        self.ctrl_manager.save(self.ctrl_file)

    def _update_operation_file(self):
        reference_year = self.reinit_year-self.ctrl_base_manager.ctrl_dict['SIMULATION_START_YEAR'] + 1
        for key in list(self.op_manager.op_dict.keys()):
            operation = self.op_manager.op_dict.pop(key)
            if key[0] > reference_year:
                self.op_manager.op_dict[(key[0] - reference_year, *key[1:])] = operation

        self.op_manager.save(self.op_file)

    def _check_is_mid_year(self):
        year_of_next_step = (self.date + timedelta(days=self.delta)).year
        return (year_of_next_step > self.reinit_year and self.date.year == self.reinit_year)


if __name__ == '__main__':
    np.random.seed(0)

    # Base argument
    env_kwargs = dict(delta=7, n_actions=11, maxN=150, start_year=1980, end_year=1980)

    # Weather shuffling
    target_year_range = np.arange(env_kwargs['start_year'], env_kwargs['end_year'] + 1)
    weather_generator_kwargs = dict(n_weather_samples=100,
                                    sampling_start_year=1980,
                                    sampling_end_year=2013,
                                    base_weather_file=CYCLES_PATH.joinpath('input', 'RockSprings.weather'),
                                    target_year_range=target_year_range)
    env_kwargs.update(dict(weather_generator_class=WeatherShuffler,
                           weather_generator_kwargs=weather_generator_kwargs))

    n_trials = 10
    np.random.seed(0)
    env = Corn(**env_kwargs)
    rewards = np.zeros(n_trials)

    for i in range(n_trials):
        s = env.reset()
        week = 0
        while True:
            a = 10 if week == 15 else 0
            s, r, done, info = env.step(a)
            rewards[i] += r
            week += 1
            if done:
                break
    print(rewards)


from cyclesgym.envs.corn import Corn
import cyclesgym.envs.observers as observers
from cyclesgym.managers import SoilNManager
from cyclesgym.envs.common import CyclesEnv
from cyclesgym.utils.paths import CYCLES_PATH
from cyclesgym.envs.common import PartialObsEnv
from cyclesgym.envs.weather_generator import WeatherShuffler, FixedWeatherGenerator

import numpy as np


class CornSoilCropWeatherObs(Corn):
    # Need to write N to output
    def __init__(self,
                 delta,
                 n_actions,
                 maxN,
                 operation_file='ContinuousCorn.operation',
                 soil_file='GenericHagerstown.soil',
                 weather_generator_class=FixedWeatherGenerator,
                 weather_generator_kwargs={
                     'base_weather_file': CYCLES_PATH.joinpath('input', 'RockSprings.weather')},
                 start_year=1980,
                 end_year=1980,
                 use_reinit=True,
                 with_obs_year=False,
                 ):
        self.rotation_size = end_year - start_year + 1
        self.use_reinit = use_reinit

        CyclesEnv.__init__(self, 
                          SIMULATION_START_YEAR=start_year,
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

        self.with_obs_year = with_obs_year
        self._post_init_setup()
        self._init_observer()
        self._generate_observation_space()
        self._generate_action_space(n_actions, maxN)
    
    # Add N manager to fields
    def _post_init_setup(self):
        super()._post_init_setup()
        self.soil_n_file = None
        self.soil_n_manager = None
    
    # Initialize soil N manager
    def _init_output_managers(self):
        super()._init_output_managers()
        self.soil_n_file = self._get_output_dir().joinpath('N.dat')
        self.soil_n_manager = SoilNManager(self.soil_n_file)
        self.output_managers.append(self.soil_n_manager)
        self.output_files.append(self.soil_n_file)
    
    # Add observer of soil to compound one
    def _init_observer(self, *args, **kwargs):
        end_year = self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR']
        self.observer = observers.compound_observer([
            observers.WeatherObserver(weather_manager=self.weather_manager, end_year=end_year),
            observers.CropObserver(crop_manager=self.crop_output_manager, end_year=end_year),
            observers.SoilNObserver(soil_n_manager=self.soil_n_manager, end_year=end_year),
            observers.NToDateObserver(end_year=end_year, with_year=self.with_obs_year)
                                           ])


def CornSoilRefined(delta, n_actions, maxN, start_year, end_year, sampling_start_year, sampling_end_year,
     n_weather_samples, fixed_weather, with_obs_year, new_holland=False):
    target_obs = ['PP', # Precipitation
                  'TX', # Max temperature
                  'TN', # Min temperature
                  'SOLAR', # Radiation
                  'RHX', # Max relative humidity
                  'RHN', # Min relative humidity
                  'STAGE', # Stage in the plant life cycle
                  'CUM. BIOMASS', # Cumulative plant biomass
                  'N STRESS', 
                  'WATER STRESS',
                  'ORG SOIL N', # The sum of microbial biomass N and stabilized soil organic N pools.
                  'PROF SOIL NO3', # Soil profile nitrate-N content.
                  'PROF SOIL NH4', # Soil profile ammonium-N content.
                  'Y', # Years left
                  'DOY' # Day of the year
                 ]

    return generate_partially_observable_env(target_obs, delta, n_actions, maxN, start_year, end_year,
                                             sampling_start_year,
                                             sampling_end_year, n_weather_samples, fixed_weather, with_obs_year,
                                             new_holland=new_holland)


def NonAdaptiveCorn(delta, n_actions, maxN, start_year, end_year, sampling_start_year, sampling_end_year,
     n_weather_samples, fixed_weather, with_obs_year, new_holland=False):
    target_obs = ['Y', # Years left
                  'DOY', # Day of the year
                  'N TO DATE'
                 ]
    return generate_partially_observable_env(target_obs, delta, n_actions, maxN, start_year, end_year, sampling_start_year,
                                      sampling_end_year, n_weather_samples, fixed_weather, with_obs_year,
                                      new_holland=new_holland)


def generate_partially_observable_env(target_obs, delta, n_actions, maxN, start_year, end_year, sampling_start_year,
                                      sampling_end_year, n_weather_samples, fixed_weather, with_obs_year,
                                      new_holland=False):
    # Weather generator
    if fixed_weather:
        weather_generator_class = FixedWeatherGenerator
        if new_holland:
            weather_generator_kwargs = {'base_weather_file': CYCLES_PATH.joinpath('input', 'RockSprings.weather')}
        else:
            weather_generator_kwargs = {'base_weather_file': CYCLES_PATH.joinpath('input', 'NewHolland.weather')}
    else:
        weather_generator_class = WeatherShuffler
        target_year_range = np.arange(start_year, end_year + 1)
        weather_generator_kwargs = dict(n_weather_samples=n_weather_samples,
                                        sampling_start_year=sampling_start_year,
                                        sampling_end_year=sampling_end_year,
                                        target_year_range=target_year_range)
        if new_holland:
            weather_generator_kwargs.update({'base_weather_file': CYCLES_PATH.joinpath('input', 'RockSprings.weather')})
        else:
            weather_generator_kwargs.update({'base_weather_file': CYCLES_PATH.joinpath('input', 'NewHolland.weather')})

    # Fully observable environment
    fully_observable_env = CornSoilCropWeatherObs(delta=delta,
                                                  n_actions=n_actions,
                                                  maxN=maxN,
                                                  start_year=start_year,
                                                  end_year=end_year,
                                                  with_obs_year=with_obs_year,
                                                  weather_generator_kwargs=weather_generator_kwargs,
                                                  weather_generator_class=weather_generator_class)

    # Mask it with target obs
    mask = compute_mask(target_obs, delta, n_actions, maxN, start_year, end_year, sampling_start_year,
                        sampling_end_year, n_weather_samples, fixed_weather, with_obs_year)
    partially_observable_env = PartialObsEnv(fully_observable_env, mask=mask)
    partially_observable_env.reset()
    return partially_observable_env


def compute_mask(target_obs,
                 delta,
                 n_actions,
                 maxN,
                 start_year,
                 end_year,
                 sampling_start_year,
                 sampling_end_year,
                 n_weather_samples,
                 fixed_weather,
                 with_obs_year):

    # Initialize environment
    if fixed_weather:
        large_obs_corn_env = CornSoilCropWeatherObs(delta=delta,
                                                    n_actions=n_actions,
                                                    maxN=maxN,
                                                    start_year=start_year,
                                                    end_year=end_year,
                                                    with_obs_year=with_obs_year)
    else:
        # TODO: Probably this part is not necessary since changing weather generator does not affect observation space
        weather_generator_kwargs = dict(n_weather_samples=n_weather_samples,
                                        sampling_start_year=sampling_start_year,
                                        sampling_end_year=sampling_end_year,
                                        target_year_range=np.arange(start_year, end_year + 1),
                                        base_weather_file=CYCLES_PATH.joinpath('input', 'RockSprings.weather'))
        large_obs_corn_env = CornSoilCropWeatherObs(delta=delta,
                                                    n_actions=n_actions,
                                                    maxN=maxN, start_year=start_year,
                                                    end_year=end_year,
                                                    with_obs_year=with_obs_year,
                                                    weather_generator_class=WeatherShuffler,
                                                    weather_generator_kwargs=weather_generator_kwargs)

    # Initialize observation space to get observation names
    s = large_obs_corn_env.reset()
    large_obs_corn_env.observer.obs_names

    # Compute mask for partially observable environment
    mask = np.isin(np.asarray(large_obs_corn_env.observer.obs_names), target_obs)
    return mask




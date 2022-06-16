from gym.envs.registration import register
from cyclesgym.envs.weather_generator import FixedWeatherGenerator, WeatherShuffler
from cyclesgym.utils.paths import CYCLES_PATH
import numpy as np


def env_name(location: str, random_weather: bool, exp_type: str, duration: str = ''):
    weather = 'RW' if random_weather else 'FW'  # Random weather or fixed weather
    if exp_type == 'fertilization':
        return f'Corn{duration}{location}{weather}-v1'
    elif exp_type == 'crop_planning':
        return f'CropPlanning{location}{weather}-v1'


def get_weather(start_year, end_year, random=False, location='RockSprings',
                sampling_start_year=1980, sampling_end_year=2005):
    if random:
        target_year_range = np.arange(start_year, end_year + 1)
        weather_generator_class = WeatherShuffler
        weather_generator_kwargs = dict(n_weather_samples=100,
                                        sampling_start_year=sampling_start_year,
                                        sampling_end_year=sampling_end_year,
                                        base_weather_file=CYCLES_PATH.joinpath(
                                            'input',  f'{location}.weather'),
                                        target_year_range=target_year_range)
    else:
        weather_generator_class = FixedWeatherGenerator
        weather_generator_kwargs = dict(base_weather_file=CYCLES_PATH.joinpath(
                                            'input',  f'{location}.weather'))
    return weather_generator_class, weather_generator_kwargs


def register_fertilization_envs():
    common_kwargs = {'delta': 7, 'n_actions': 11, 'maxN': 150, 'start_year': 1980}
    durations = [(1, 'Short'), (2, 'Middle'), (5, 'Long')]
    locations = ['RockSprings', 'NewHolland']

    # Loop through duration, random vs fixed weather, location
    for rw in [True, False]:
        for l in locations:
            for d, d_name in durations:
                # Get env name
                name = env_name(duration=d_name, location=l, random_weather=rw, exp_type='fertilization')

                # Copy common kwargs and update using duration and weather
                kwargs = common_kwargs.copy()

                start_year = kwargs['start_year']
                end_year = start_year + (d - 1)

                weather_generator_class, weather_generator_kwargs = \
                    get_weather(start_year, end_year, random=rw, location=l)

                kwargs.update(dict(end_year=end_year,
                                   weather_generator_class=weather_generator_class,
                                   weather_generator_kwargs=weather_generator_kwargs))
                register(
                    id=name,
                    entry_point='cyclesgym.envs:Corn',
                    kwargs=kwargs
                )


def register_crop_planning_envs():
    start_year = 1980
    end_year = 1998
    common_kwargs = dict(start_year=start_year, end_year=end_year,
                         rotation_crops=['CornSilageRM.90', 'SoybeanMG.3'])
    locations = ['RockSprings', 'NewHolland']

    # Loop through random vs fixed weather, location
    for rw in [True, False]:
        for l in locations:
            name = env_name(location=l, random_weather=rw, exp_type='crop_planning')

            kwargs = common_kwargs.copy()

            weather_generator_class, weather_generator_kwargs = \
                get_weather(start_year, end_year, random=rw, location=l,
                            sampling_start_year=start_year,
                            sampling_end_year=end_year)

            kwargs.update(dict(end_year=end_year,
                               weather_generator_class=weather_generator_class,
                               weather_generator_kwargs=weather_generator_kwargs))
            print(kwargs)
            register(
                id=name,
                entry_point='cyclesgym.envs:CropPlanningFixedPlanting',
                kwargs=kwargs
            )


register_fertilization_envs()
register_crop_planning_envs()

import pandas as pd
from typing import Sequence, List
import random
import calendar
import numpy as np
from pathlib import Path


from cyclesgym.managers import WeatherManager
from cyclesgym.envs.utils import MyTemporaryDirectory

__all__ = ['shuffle_weather', 'adapt_weather_year', 'generate_random_weather']


def shuffle_weather(weather_manager: WeatherManager,
                    duration: int,
                    n_samples: int = 1) -> List[WeatherManager]:
    """
    Sample random weather subsequnces of given duration from base manager.

    Given a weather manager containing weather conditions for years from x to
    y, we first sample an initial year from this range, then select all the
    years in [initial_year, initial_year + duration], and shuffle them around.
    We repeat this procedure to generate n_samples of shuffled weather.

    Parameters
    ----------
    weather_manager: WeatherManager
        Weatther manager containing the original weather
    duration: int
        Duration in years of the intervals to subsample at random
    n_samples: int
        Number of different weather samples to collect

    Returns
    -------
    shuffled_weather: List[WeatherManager]
        List of weather managers containing the shuffled weather years of
        specific duration

    """

    # Extract mutables and immutables data frames
    imm_df = weather_manager.immutables
    mutables = weather_manager.mutables

    # Get years
    years = np.asarray(mutables['YEAR'].unique())

    # Get years that can be used as a starting point for the sequence
    valid_start_years = years[years + (duration - 1) <= np.max(years)]
    grouped_by_year = list(mutables.groupby(by='YEAR'))
    shuffled_weather = []

    for _ in range(n_samples):
        # Sample starting year
        start_year = np.random.choice(valid_start_years)

        # Shuffled sequence of  years in [start_year, start_year + duration]
        sampled_years = np.arange(start_year, start_year + duration)
        random.shuffle(sampled_years)

        # Concatenates the dfs corresponding to sampled years
        # (assumes grouped by year is sorted)
        new_mutables_df = pd.concat(
            [grouped_by_year[y - years.min()][1] for y in sampled_years],
            ignore_index=True)

        # Create weather managers
        new_weather = WeatherManager.from_df(immutables_df=imm_df,
                                             mutables_df=new_mutables_df)
        shuffled_weather.append(new_weather)
    return shuffled_weather


def adapt_weather_year(weather_manager: WeatherManager,
                       target_year_range: Sequence[int]):
    """
    Set the years for the weather contained in the weather manager.

    Reset the years in the weather file to those specified in
    target_year_range. When resetting, if the original year is leap and the
    target one is not, we drop the last day of the year. If the original year
    is not leap and the target one is, we add one day at the end of the year
    using the average of the last 7 days.

    Parameters
    ----------
    weather_manager: WeatherManager
        Manager containing the weather whose years we adjust
    target_year_range: Sequence[int]
        Sequence of years that we want to set for the manager (must be of the
        same length as number of years in the manager).
    """
    # Extract mutables and immutables data frames
    imm_df = weather_manager.immutables
    mutables = weather_manager.mutables

    # Read years in input
    original_years = np.asarray(mutables['YEAR'].unique())
    grouped_by_year = list(mutables.groupby(by='YEAR'))
    target_year_range = np.asarray(target_year_range)

    # Validate input
    if not target_year_range.size == original_years.size:
        print(target_year_range)
        print(original_years)
        raise ValueError(f'Target year range should be of the same size as '
                         f'original years ({original_years.size}). It is of'
                         f'size {target_year_range.size} instead')

    for (o_y, o_y_df), t_y in zip(grouped_by_year, target_year_range):

        # If turning a leap year into a non-leap one, remove last day
        if calendar.isleap(o_y) and not calendar.isleap(t_y):
            o_y_df.drop(o_y_df.tail(1).index, inplace=True)

        # If turning a non-leap year into a leap one, add one day that is the
        # avg of the year's last week
        elif not calendar.isleap(o_y) and calendar.isleap(t_y):
            last_week_average = o_y_df.iloc[-7:, :].mean(axis=0)
            last_week_average.loc['DOY'] = 366

            new_ind = o_y_df.index.values.max() + 1
            o_y_df.loc[new_ind, :] = last_week_average

        # Set target year
        o_y_df['YEAR'] = t_y

    # TODO: This can probably be done in a much smarter way
    # Cast doy as integer because intepreted as float in last_week_average
    new_grouped_by_year = []

    for year, year_df in grouped_by_year:
        year_df = year_df.astype({'YEAR': int, 'DOY': int})
        new_grouped_by_year.append((year, year_df))

    # Create mutable weather df from  adapted years
    adapted_weather_df = pd.concat(
        [element[1] for element in new_grouped_by_year], ignore_index=True)

    return WeatherManager.from_df(immutables_df=imm_df,
                                  mutables_df=adapted_weather_df)


def generate_random_weather(weather_manager: WeatherManager,
                            duration: int,
                            target_year_range: Sequence[int],
                            n_samples: int = 1):
    # TODO: To improve merging the two functions in one
    shuffled_weather = shuffle_weather(weather_manager=weather_manager,
                                       duration=duration,
                                       n_samples=n_samples)
    new_weather = []
    for manager in shuffled_weather:
        new_weather.append(adapt_weather_year(manager, target_year_range))
    return new_weather


if __name__ == '__main__':
    from cyclesgym.paths import CYCLES_PATH
    import time

    # Load base weather data
    fname = CYCLES_PATH.joinpath('input', 'RockSprings.weather')
    manager = WeatherManager(fname)

    t = time.time()

    # Create weather
    new_weather = generate_random_weather(weather_manager=manager,
                                          duration=10,
                                          target_year_range=np.arange(1980, 1990),
                                          n_samples=50)

    directory = MyTemporaryDirectory(Path().cwd().joinpath('tmp'))

    for i, w in enumerate(new_weather):

        # Save new weather
        w.save(directory.name.joinpath(f'{i}.weather'))
    print(f'elapsed time {time.time() - t}')

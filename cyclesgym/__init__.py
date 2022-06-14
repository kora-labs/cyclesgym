from gym.envs.registration import register


def env_name(location: str, random_weather: bool, exp_type: str, duration: str = ''):
    weather = 'RW' if random_weather else 'FW'  # Random weather or fixed weather
    if exp_type == 'fertilization':
        return f'Corn{duration}{location}{weather}-v1'
    elif exp_type == 'crop_planning':
        return f'CropPlanning{location}{weather}-v1'

def register_fertilization_envs():
    common_kwargs = {'delta': 7, 'n_actions': 11, 'maxN': 150, 'start_year': 1980}
    durations = [(1, 'Short'), (2, 'Middle'), (5, 'Long')]
    locations = ['RockSprings', 'NewHolland']

    # Loop through duration, random vs fixed weather, location
    for rw in [True, False]:
        for l in locations:
            for d, d_name in durations:
                name = env_name(duration=d_name, location=l, random_weather=rw, exp_type='fertilization')
                kwargs = common_kwargs.copy()
                end_year = kwargs['start_year'] + (d - 1)
                kwargs.update({'end_year': end_year})
                register(
                    id=name,
                    entry_point='cyclesgym.envs:Corn',
                    kwargs=kwargs
                )


def register_crop_planning_envs():
    locations = ['RockSprings', 'NewHolland']

    # Loop through random vs fixed weather, location
    for rw in [True, False]:
        for l in locations:
            name = env_name(location=l, random_weather=rw, exp_type='crop_planning')
            # Should register here


register_fertilization_envs()
register_crop_planning_envs()

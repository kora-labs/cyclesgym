from cyclesgym.envs import Corn
from cyclesgym.utils import compare_env, maximum_absolute_percentage_error

if __name__ == '__main__':
    START_YEAR = 1980
    END_YEAR = 1983

    delta = 7
    n_actions = 7
    maxN = 120

    env = Corn(delta=delta, n_actions=n_actions, maxN=maxN, start_year=START_YEAR,
               end_year=END_YEAR, use_reinit=False)

    env_modified_soil = Corn(delta=delta, n_actions=n_actions, maxN=maxN, start_year=START_YEAR,
                             end_year=END_YEAR, soil_file='GenericHagerstown_modified.soil', use_reinit=False)

    obs_cont, obs_impr, time_cont, time_impr = compare_env(env, env_modified_soil)
    print(f'Time of continuous environemnt over {END_YEAR - START_YEAR} years: {time_cont}')
    print(f'Time of improved environemnt over {END_YEAR - START_YEAR} years: {time_impr}')

    max_ape = maximum_absolute_percentage_error(obs_cont, obs_impr)
    print(f'Maximum percentage error: {max_ape} %')

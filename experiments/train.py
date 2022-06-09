import os.path
import time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from cyclesgym.utils import EvalCallbackCustom
from pathlib import Path
from eval import _evaluate_policy
import gym
from cyclesgym.envs.corn import Corn
from cyclesgym.envs.crop_planning import CropPlanning, CropPlanningFixedPlanting, CropPlanningFixedPlantingRandomWeather
from cyclesgym.envs.crop_planning import CropPlanningFixedPlantingRandomWeatherRotationObserver, CropPlanningFixedPlantingRotationObserver
import wandb
from wandb.integration.sb3 import WandbCallback
import sys
import random
import argparse
from cyclesgym.paths import CYCLES_PATH


class Train:
    """ Trainer object to wrap model training and handle environment creation, evaluation """

    def __init__(self, experiment_config) -> None:
        self.config = experiment_config
        # rl config is configured from wandb config

    def create_envs(self):
        eval_env_train = self.env_maker(start_year=self.config['train_start_year'],
                                        end_year=self.config['train_end_year'],
                                        training=False,
                                        env_class=self.config['eval_env_class'])

        eval_env_new_years = self.env_maker(start_year=self.config['eval_start_year'],
                                            end_year=self.config['eval_end_year'],
                                            training=False,
                                            env_class=self.config['eval_env_class'])

        eval_env_other_loc = self.env_maker(start_year=self.config['train_start_year'],
                                            end_year=self.config['train_end_year'],
                                            weather_file='NewHolland.weather',
                                            training=False,
                                            env_class=self.config['eval_env_class'])

        eval_env_other_loc_long = self.env_maker(start_year=self.config['train_start_year'],
                                                 end_year=self.config['eval_end_year'] - 1,
                                                 weather_file='NewHolland.weather', \
                                                 env_class=self.config['eval_env_class'],
                                                 training=False)

        return [eval_env_train, eval_env_new_years, eval_env_other_loc, eval_env_other_loc_long]

    def env_maker(self, env_class=CropPlanningFixedPlanting,
                  training=True, n_procs=4, start_year=1980, end_year=2000, soil_file='GenericHagerstown.soil',
                  weather_file='RockSprings.weather', n_weather_samples=None):
        if not training:
            n_procs = 1

        if isinstance(env_class, str):
            env_class = globals()[env_class]

        def make_env():
            # creates a function returning the basic env. Used by SubprocVecEnv later to create a
            # vectorized environment
            def _f():
                env_conf = dict(start_year=start_year, end_year=end_year, soil_file=soil_file,
                                weather_file=weather_file, rotation_crops=['CornRM.100', 'SoybeanMG.3'])
                if n_weather_samples is not None:
                    env_conf['n_weather_samples'] = n_weather_samples

                env = env_class(**env_conf)

                env = gym.wrappers.RecordEpisodeStatistics(env)
                return env

            return _f

        env = SubprocVecEnv([make_env() for i in range(n_procs)], start_method='fork')
        env = VecMonitor(env)
        norm_reward = (training and self.config['norm_reward'])
        env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=5000., clip_reward=5000.)
        return env

    def create_callback(self, model_dir):
        eval_freq = int(self.config['eval_freq'] / self.config['n_process'])

        [eval_env_train, eval_env_new_years, eval_env_other_loc, eval_env_other_loc_long] = self.create_envs()
        def get_callback(env, suffix, deterministic):
            return EvalCallbackCustom(env, best_model_save_path=str(model_dir.joinpath(suffix)),
                                      log_path=str(model_dir.joinpath(suffix)), eval_freq=eval_freq,
                                      deterministic=deterministic, render=False, eval_prefix=suffix)

        eval_callback_det = get_callback(eval_env_train, 'eval_det', True)
        eval_callback_sto = get_callback(eval_env_train, 'eval_sto', False)

        eval_callback_det_new_years = get_callback(eval_env_new_years, 'eval_det_new_years', True)
        eval_callback_sto_new_years = get_callback(eval_env_new_years, 'eval_sto_new_years', False)

        eval_callback_det_other_loc = get_callback(eval_env_other_loc, 'eval_det_other_loc', True)
        eval_callback_sto_other_loc = get_callback(eval_env_other_loc, 'eval_sto_other_loc', False)

        eval_callback_det_other_loc_long = get_callback(eval_env_other_loc_long, 'eval_det_other_loc_long', True)
        eval_callback_sto_other_loc_long = get_callback(eval_env_other_loc_long, 'eval_sto_other_loc_long', False)

        return [eval_callback_det, eval_callback_sto, eval_callback_det_new_years, eval_callback_sto_new_years,
                eval_callback_det_other_loc, eval_callback_sto_other_loc, eval_callback_det_other_loc_long,
                eval_callback_sto_other_loc_long]

    def train(self):
        train_env = self.env_maker(start_year=self.config['train_start_year'],
                                   end_year=self.config['train_end_year'],
                                   env_class=self.config['env_class'],
                                   training=True, n_procs=self.config['n_process'],
                                   n_weather_samples=self.config['n_weather_samples'])
        dir = wandb.run.dir
        model_dir = Path(dir).joinpath('models')

        if self.config["method"] == "A2C":
            model = A2C('MlpPolicy', train_env, verbose=self.config['verbose'], tensorboard_log=dir,
                        device=self.config['device'])
        elif self.config["method"] == "PPO":
            model = PPO('MlpPolicy', train_env, n_steps=self.config['n_steps'], batch_size=self.config['batch_size'],
                        n_epochs=self.config['n_epochs'], verbose=self.config['verbose'], tensorboard_log=dir,
                        device=self.config['device'])
        elif self.config["method"] == "DQN":
            model = DQN('MlpPolicy', train_env, verbose=self.config['verbose'], tensorboard_log=dir,
                        device=self.config['device'])
        else:
            raise Exception("Not an RL method that has been implemented")

        # The test environment will automatically have the same observation normalization applied to it by
        # EvalCallBack

        callback = self.create_callback(model_dir)

        callback = [WandbCallback(model_save_path=str(model_dir),
                                  model_save_freq=int(self.config['eval_freq'] / self.config['n_process']))] + callback
        model.learn(total_timesteps=self.config["total_timesteps"], callback=callback)

        return model, eval_env

    def evaluate_log(self, model, eval_env):
        """
        Runs policy deterministically (1 episode) and stochastically (5 episodes)
        logs the fertilization actions taken by the model

        Parameters
        ----------
        model: trained agent
        eval_env

        Returns
        -------
        mean deterministic reward

        """
        mean_r_det, _, actions_det, episode_rewards_det = _evaluate_policy(model,
                                                                           env=eval_env,
                                                                           n_eval_episodes=1,
                                                                           deterministic=True)
        mean_r_stoc, std_r_stoc, actions_stoc, episode_rewards_stoc = _evaluate_policy(model,
                                                                                       env=eval_env,
                                                                                       n_eval_episodes=5,
                                                                                       deterministic=False)
        wandb.log({'deterministic_return': mean_r_det,
                   'stochastic_return_mean': mean_r_stoc,
                   'stochastic_return_std': std_r_stoc,
                   })
        episode_actions_names = [*list(f"det{i + 1}" for i in range(len(actions_det))),
                                 *list(f"stoc{i + 1}" for i in range(len(actions_stoc)))]
        episode_actions = [*actions_det, *actions_stoc]
        fertilizer_table = wandb.Table(
            columns=['Run', 'Total Fertilizer', *[f'Week{i}' for i in range(53)]])
        for i in range(len(episode_actions)):
            acts = episode_actions[i]
            data = [[week, fert] for (week, fert) in zip(range(53), acts)]
            table = wandb.Table(data=data, columns=['Week', 'N added'])
            fertilizer_table.add_data(
                *[episode_actions_names[i], np.sum(acts), *acts])
            wandb.log({f'train/actions/{episode_actions_names[i]}':
                           wandb.plot.bar(table, 'Week', 'N added',
                                          title=f'Training action sequence {episode_actions_names[i]}')})
        wandb.log({'train/fertilizer': fertilizer_table})
        return mean_r_det


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-fw', '--fixed_weather', default=False,
                        help='Whether to use a fixed weather')
    parser.add_argument('-na', '--non_adaptive', default=False,
                        help='Whether to use a non-adaptive policy (observation space being only a trailing window of'
                             'the crop rotation used so far')
    parser.add_argument('-s', '--seed', type=int, default=0, metavar='N',
                        help='The random seed used for all number generators')

    args = vars(parser.parse_args())

    set_random_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    print(args)
    print(CYCLES_PATH, os.path.isfile(CYCLES_PATH))
    if args['fixed_weather'] == 'True':
        env_class = 'CropPlanningFixedPlanting'
        n_weather_samples = None
        eval_env_class = 'CropPlanningFixedPlanting'
        if args['non_adaptive'] == 'True':
            env_class = 'CropPlanningFixedPlantingRotationObserver'
            eval_env_class = 'CropPlanningFixedPlantingRotationObserver'
    else:
        env_class = 'CropPlanningFixedPlantingRandomWeather'
        n_weather_samples = 1000
        eval_env_class = 'CropPlanningFixedPlanting'
        if args['non_adaptive'] == 'True':
            env_class = 'CropPlanningFixedPlantingRandomWeatherRotationObserver'
            eval_env_class = 'CropPlanningFixedPlantingRotationObserver'

    config = dict(train_start_year=1980, train_end_year=1998, eval_start_year=1998, eval_end_year=2016,
                  total_timesteps=1000000, eval_freq=1000, n_steps=80, batch_size=64, n_epochs=10, run_id=0,
                  norm_reward=True, method="PPO", verbose=1, n_process=8, device='auto',
                  env_class=env_class, eval_env_class=eval_env_class, n_weather_samples=n_weather_samples)

    config.update(args)

    wandb.init(
        config=config,
        sync_tensorboard=True,
        project="experiments_crop_planning",
        entity='koralabs',
        monitor_gym=True,  # automatically upload gym environements' videos
        save_code=True,
    )

    config = wandb.config

    print(config)

    trainer = Train(config)
    model, eval_env = trainer.train()
    # Load the saved statistics

    eval_env = VecNormalize.load(config['stats_path'], eval_env)
    #  do not update moving averages at test time
    eval_env.training = False
    # reward normalization is not needed at test time
    eval_env.norm_reward = False

    #trainer.evaluate_log(model, eval_env)


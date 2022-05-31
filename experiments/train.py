import time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
# from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from cyclesgym.utils import EvalCallbackCustom
from pathlib import Path
from eval import _evaluate_policy
import gym
from cyclesgym.envs.corn import Corn
from cyclesgym.envs.crop_planning import CropPlanning, CropPlanningFixedPlanting
import wandb
from wandb.integration.sb3 import WandbCallback
import sys


class Train:
    """ Trainer object to wrap model training and handle environment creation, evaluation """

    def __init__(self, experiment_config) -> None:
        self.config = experiment_config
        # rl config is configured from wandb config

    def env_maker(self, training=True, n_procs=4, start_year=1980, end_year=2000, soil_file='GenericHagerstown.soil',
                  weather_file='RockSprings.weather'):
        if not training:
            n_procs = 1

        def make_env():
            # creates a function returning the basic env. Used by SubprocVecEnv later to create a
            # vectorized environment
            def _f():
                env = CropPlanningFixedPlanting(start_year=start_year, end_year=end_year, soil_file=soil_file,
                                                weather_file=weather_file, rotation_crops=['CornRM.100', 'SoybeanMG.3'])
                env = gym.wrappers.RecordEpisodeStatistics(env)
                return env

            return _f

        env = SubprocVecEnv([make_env() for i in range(n_procs)], start_method='fork')

        norm_reward = (training and self.config['norm_reward'])
        env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=5000., clip_reward=5000.)

        env = VecMonitor(env)
        return env

    def train(self):
        train_env = self.env_maker(start_year=self.config['train_start_year'],
                                   end_year=self.config['train_end_year'],
                                   training=True, n_procs=self.config['n_process'])
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
        eval_freq = int(self.config['eval_freq'] / self.config['n_process'])
        eval_env_train = self.env_maker(start_year=self.config['train_start_year'],
                                        end_year=self.config['train_end_year'],
                                        training=False)

        eval_env_new_years = self.env_maker(start_year=self.config['eval_start_year'],
                                            end_year=self.config['eval_end_year'],
                                            training=False)

        eval_env_other_loc = self.env_maker(start_year=self.config['train_start_year'],
                                            end_year=self.config['eval_end_year']-1,
                                            weather_file='NewHolland.weather',
                                            training=False)

        eval_callback_det = EvalCallbackCustom(eval_env_train, best_model_save_path=str(model_dir.joinpath('eval_det')),
                                         log_path=str(model_dir.joinpath('eval_det')),
                                         eval_freq=eval_freq, deterministic=True, render=False,
                                               eval_prefix='eval_det')
        eval_callback_sto = EvalCallbackCustom(eval_env_train, best_model_save_path=str(model_dir.joinpath('eval_sto')),
                                         log_path=str(model_dir.joinpath('eval_sto')),
                                         eval_freq=int(self.config['eval_freq'] / self.config['n_process']),
                                         deterministic=False, render=False,
                                               eval_prefix='eval_sto')

        eval_callback_det_new_years = EvalCallbackCustom(eval_env_new_years,
                                                   best_model_save_path=str(model_dir.joinpath('eval_det_new_years')),
                                                   log_path=str(model_dir.joinpath('eval_det_new_years')),
                                                   eval_freq=eval_freq, deterministic=True, render=False,
                                                         eval_prefix='eval_det_new_years'
                                                         )
        eval_callback_sto_new_years = EvalCallbackCustom(eval_env_new_years,
                                                   best_model_save_path=str(model_dir.joinpath('eval_sto_new_years')),
                                                   log_path=str(model_dir.joinpath('eval_sto_new_years')),
                                                   eval_freq=eval_freq, deterministic=False,
                                                   render=False,
                                                         eval_prefix='eval_sto_new_years')

        eval_callback_det_other_loc = EvalCallbackCustom(eval_env_other_loc,
                                                   best_model_save_path=str(model_dir.joinpath('eval_det_other_loc')),
                                                   log_path=str(model_dir.joinpath('eval_det_other_loc')),
                                                   eval_freq=eval_freq, deterministic=True, render=False,
                                                         eval_prefix='eval_det_other_loc')
        eval_callback_sto_other_loc = EvalCallbackCustom(eval_env_other_loc,
                                                   best_model_save_path=str(model_dir.joinpath('eval_sto_other_loc')),
                                                   log_path=str(model_dir.joinpath('eval_sto_other_loc')),
                                                   eval_freq=eval_freq, deterministic=False,
                                                   render=False, eval_prefix='eval_sto_other_loc')

        callback = [WandbCallback(model_save_path=str(model_dir),
                                  model_save_freq=int(self.config['eval_freq'] / self.config['n_process'])),
                    eval_callback_det, eval_callback_sto, eval_callback_det_new_years, eval_callback_sto_new_years,
                    eval_callback_det_other_loc, eval_callback_sto_other_loc]
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

    if (len(sys.argv) > 1):
        method = str(sys.argv[1])
    else:
        method = "A2C"

    config = dict(train_start_year=1980, train_end_year=1998, eval_start_year=1998, eval_end_year=2016,
                  total_timesteps=1000000, eval_freq=1000, n_steps=80, batch_size=64, n_epochs=10, run_id=0,
                  norm_reward=True, method="PPO", verbose=1, n_process=8, device='cpu')

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


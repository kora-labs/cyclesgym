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
from stable_baselines3.common.callbacks import EvalCallback

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

    def env_maker(self, training=True, n_procs=4):
        if not training:
            n_procs = 1

        def make_env():
            # creates a function returning the basic env. Used by SubprocVecEnv later to create a
            # vectorized environment
            def _f():
                env = CropPlanningFixedPlanting(start_year=1980, end_year=2000,
                                                rotation_crops=['CornRM.100', 'SoybeanMG.3'])
                env = gym.wrappers.RecordEpisodeStatistics(env)
                return env

            return _f

        env = SubprocVecEnv([make_env() for i in range(n_procs)], start_method='fork')

        norm_reward = (training and self.config['norm_reward'])
        env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=5000., clip_reward=5000.)

        env = VecMonitor(env)
        return env

    def train(self):
        train_env = self.env_maker(training=True, n_procs=config['n_process'])
        if config["method"] == "A2C":
            model = A2C('MlpPolicy', train_env, verbose=config['verbose'], tensorboard_log=f"runs",
                        device=config['device'])
        elif config["method"] == "PPO":
            model = PPO('MlpPolicy', train_env, n_steps=config['n_steps'], batch_size=config['batch_size'],
                        n_epochs=config['n_epochs'], verbose=config['verbose'], tensorboard_log=f"runs",
                        device=config['device'])
        elif config["method"] == "DQN":
            model = DQN('MlpPolicy', train_env, verbose=config['verbose'], tensorboard_log=f"runs",
                        device=config['device'])
        else:
            raise Exception("Not an RL method that has been implemented")

        # The test environment will automatically have the same observation normalization applied to it by
        # EvalCallBack
        eval_env = self.env_maker(training=False)
        eval_callback_det = EvalCallback(eval_env, best_model_save_path='./logs/',
                                         log_path='runs', eval_freq=config['eval_freq'],
                                         deterministic=True, render=False)
        eval_callback_sto = EvalCallback(eval_env, best_model_save_path='./logs/',
                                         log_path='runs', eval_freq=config['eval_freq'],
                                         deterministic=False, render=False)
        callback = [WandbCallback(), eval_callback_det, eval_callback_sto]
        model.learn(total_timesteps=self.config["total_timesteps"], callback=callback)
        model.save(str(self.config['run_id']) + '.zip')
        train_env.save(self.config['stats_path'])
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

    config = dict(total_timesteps=100000, eval_freq=1000, n_steps=20, batch_size=64, n_epochs=10, run_id=0,
                  norm_reward=True, stats_path='runs/vec_normalize.pkl',
                  method="PPO", verbose=1, n_process=1, device='auto')

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


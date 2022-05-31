import matplotlib.pyplot as plt
import time
import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


import os
from typing import Optional, Union

import gym
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization


eps = 1e-8


def maximum_absolute_percentage_error(y1, y2):
    return 100*np.max(np.abs(y1-y2)/np.abs(y1+eps))


def mean_absolute_percentage_error(y1, y2):
    return 100*np.mean(np.abs(y1-y2)/np.abs(y1+eps))


def plot_two_environments(df1, df2, labels, columns):
    for col in columns:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        original = df1.iloc[:, col]
        new = df2.iloc[:, col]
        ax1.plot(original, label=labels[0], marker='x')
        ax1.plot(new, label=labels[1], marker='o', mfc='none')
        ax2.plot((100 * (new - original) / (original + 1e-5)), label='Percentage difference', c='green')
        ax1.set_title(df1.columns[col])
        fig.legend()
        plt.show()


def run_env(env, actions_to_use=None):
    i = 0
    observations = []
    actions = []
    while True:
        if actions_to_use is None:
            a = env.action_space.sample()
        else:
            a = actions_to_use[i]

        s, r, done, info = env.step(a)
        observations.append(s)
        actions.append(a)

        if done:
            break
        i = i + 1

    return observations, actions


def compare_env(env1, env2):
    env1.reset()
    env2.reset()

    t = time.time()
    obs_1, actions = run_env(env1)
    time_1 = time.time() - t

    t = time.time()
    obs_2, _ = run_env(env2, actions)
    time_2 = time.time() - t

    obs_1 = np.array(obs_1)
    obs_2 = np.array(obs_2)

    df1 = pd.DataFrame(obs_1)
    df2 = pd.DataFrame(obs_2)
    plot_two_environments(df1, df2, ['1', '2'], range(0, obs_1.shape[1]))

    return obs_1, obs_2, time_1, time_2


def diff_pd(df1, df2):
    """Identify differences between two pandas DataFrames"""
    assert (df1.columns == df2.columns).all(), \
        "DataFrame column names are different"
    if any(df1.dtypes != df2.dtypes):
        "Data Types are different, trying to convert"
        df2 = df2.astype(df1.dtypes)
    if df1.equals(df2):
        return None
    else:
        # need to account for np.nan != np.nan returning True
        diff_mask = (df1 != df2) & ~(df1.isnull() & df2.isnull())
        ne_stacked = diff_mask.stack()
        changed = ne_stacked[ne_stacked]
        changed.index.names = ['id', 'col']
        difference_locations = np.where(diff_mask)
        changed_from = df1.values[difference_locations]
        changed_to = df2.values[difference_locations]
        return pd.DataFrame({'from': changed_from, 'to': changed_to},
                            index=changed.index)
    

class EvalCallbackCustom(EvalCallback):

    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            callback_on_new_best: Optional[BaseCallback] = None,
            callback_after_eval: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
            eval_prefix: str = 'eval'
    ):
        super(EvalCallbackCustom, self).__init__(eval_env=eval_env,
                                                 callback_on_new_best=callback_on_new_best,
                                                 callback_after_eval=callback_after_eval,
                                                 n_eval_episodes=n_eval_episodes,
                                                 eval_freq=eval_freq,
                                                 log_path=log_path,
                                                 best_model_save_path=best_model_save_path,
                                                 deterministic=deterministic,
                                                 render=render,
                                                 verbose=verbose,
                                                 warn=warn)
        self.eval_prefix = eval_prefix

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(self.eval_prefix + "/mean_reward", float(mean_reward))
            self.logger.record(self.eval_prefix + "/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record(self.eval_prefix + "/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(self.eval_prefix + "/time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
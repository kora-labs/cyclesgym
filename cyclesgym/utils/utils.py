import matplotlib.pyplot as plt
import time
import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os
from stable_baselines3.common.evaluation import evaluate_policy
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gym
import numpy as np
from numpy import ndarray
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization
from stable_baselines3.common.policies import obs_as_tensor

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
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
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
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True


def predict_proba(model, obs):
    obs = obs_as_tensor(obs, model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = [d.probs.detach().numpy() for d in dis.distribution]
    return probs


def _evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Tuple[List[Any], List[Any], ndarray, List[Any]]:
    """
    Modified version of stablebaselines evaluate_policy => returns the actions taken by the agent
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode, episode actions, episode rewards.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_actions = []
    episode_action_rewards = []
    episode_probs = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    # current_actions = np.array([])
    current_actions = []
    current_probs = []

    # current_actions = np.empty((episode_count_targets, n_envs))
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, episode_start=episode_starts,
                                        deterministic=deterministic)
        probs = predict_proba(model, observations)
        current_actions.append(actions)
        current_probs.append(probs)
        # current_actions = np.append(current_actions, actions)
        # current_actions[episode_counts] = actions
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                episode_action_rewards.append(rewards[i])
                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    actions_i = np.array(current_actions)[:, i]
                    probs_i = current_probs
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            episode_actions.append(actions_i)
                            episode_probs.append(probs_i)
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_actions.append(actions_i)
                        episode_probs.append(probs_i)
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    current_actions = []

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    episode_actions = np.array(episode_actions)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, \
            "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_actions, episode_rewards
    return mean_reward, std_reward, episode_actions, episode_rewards, episode_probs, episode_action_rewards

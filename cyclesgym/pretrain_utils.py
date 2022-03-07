import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from pathlib import Path
from imitation.algorithms.bc import BC
from cyclesgym.env import CornEnv, PartialObsCornEnv
from cyclesgym.dummy_policies import OpenLoopPolicy
import shutil
import torch as th
import time
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.data import types, rollout
from cyclesgym.paths import PROJECT_PATH

import warnings
warnings.filterwarnings("ignore")

N_ACTIONS = 3
DELTA_T = 14
N_DAYS = 366


def collect_expert_trajectories(env, expert_policy=None, path=None,
                                rollout_episodes=10):
    if expert_policy is None:
        expert_policy = PPO('MlpPolicy', env)
        expert_policy.learn(5000)

    if path is None:
        path = Path().cwd().joinpath('tmp', 'expert_trajectories.pkl')

    if path.suffix != '.pkl':
        raise ValueError('A valid path ending in .pkl should be provided to store the trajectories')

    # Make sure parent dir exists
    path.parents[0].mkdir(exist_ok=True, parents=True)

    sample_until = rollout.make_sample_until(min_episodes=rollout_episodes,
                                             min_timesteps=None)
    rollout.rollout_and_save(str(path), expert_policy, env, sample_until,
                             unwrap=False)
    return path


def train_bc(env, path, policy, epochs=10, **bc_kwargs):
    expert_trajs = types.load(path)
    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=policy,
        demonstrations=rollout.flatten_trajectories(expert_trajs),
        **bc_kwargs
        )
    bc_trainer.train(n_epochs=epochs)
    return bc_trainer.policy


def evaluate(policy, env, n_episodes, render=False):
    for i in range(n_episodes):
        print(f'starting {i}th episode')
        done = False
        obs = env.reset()
        R = 0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, r, done, info = env.step(action)
            R += r
            if render:
                env.render()
        print(f'Final return {R}')


def env_creator():
    mask = np.zeros(26, dtype=bool)
    mask[-2:] = True
    return Monitor(PartialObsCornEnv('ContinuousCorn.ctrl',
                                      n_actions=N_ACTIONS,
                                      delta=DELTA_T,
                                     mask=mask))


def example(behavioral_cloning=False):
    n_steps = np.ceil(N_DAYS / DELTA_T).astype(int)
    expert_policy_action_step = np.ceil(110/DELTA_T).astype(int) - 1
    action_sequence = np.zeros(n_steps, dtype=int)
    action_sequence[expert_policy_action_step] = N_ACTIONS - 1
    expert_policy = OpenLoopPolicy(action_sequence)

    env = DummyVecEnv([env_creator])
    path = collect_expert_trajectories(env,
                                       path=None,
                                       expert_policy=expert_policy,
                                       rollout_episodes=10)
    bc_kwargs = dict(ent_weight=0.5, l2_weight=0)

    env = DummyVecEnv([env_creator])
    training_env = SubprocVecEnv([env_creator] * 4)
    model = PPO('MlpPolicy', training_env, verbose=0, learning_rate=2e-4,
                n_steps=n_steps * 4, batch_size=n_steps * 4,
                n_epochs=10, ent_coef=1e-4, policy_kwargs={'net_arch': [16]})

    if behavioral_cloning:
        train_bc(env, path, model.policy, epochs=40, **bc_kwargs)

    observations = np.empty((n_steps, env.envs[0].observation_space.shape[0]))
    dummy_env = env_creator()
    observations[0, :] = dummy_env.reset()
    for i in range(1, n_steps):
        action = N_ACTIONS - 1 if i == expert_policy_action_step else 0
        observations[i, :], r, done, info = dummy_env.step(action)

    probs = get_action_distributions(observations, model.policy)
    plt.figure()
    plt.plot(probs)
    plt.title(f'Action distribution after BC')
    plt.legend([f'{i*120 / (N_ACTIONS-1)}' for i in range(N_ACTIONS)])
    plt.show()


    print('Expert policy evaluation')
    evaluate(expert_policy, env_creator(), 1)
    print('Student policy evaluation')
    mean_r, _ = evaluate_policy(model, env=DummyVecEnv([env_creator]),
                                     n_eval_episodes=5, deterministic=False)
    print(mean_r)

    print('Student policy evaluation after additional training')
    t = time.time()
    model.learn(total_timesteps=int(5e3))
    print(f'Elapsed time {time.time() - t}')

    probs = get_action_distributions(observations, model.policy)
    plt.figure()
    plt.plot(probs)
    plt.title(f'Action distribution after RL training')
    plt.legend([f'{i*120 / (N_ACTIONS-1)}' for i in range(N_ACTIONS)])

    plt.show()

    plt.plot(env.envs[0].episode_returns)
    mean_r, std_r, = evaluate_policy(model, env=DummyVecEnv([env_creator]),
                                     n_eval_episodes=5, deterministic=False)
    print(f'Non deterministic return {mean_r} +- {std_r}')
    mean_r, std_r, = evaluate_policy(model, env=DummyVecEnv([env_creator]),
                                     n_eval_episodes=5, deterministic=True)
    print(f'Deterministic return {mean_r} +- {std_r}')
    # evaluate(model.policy, env_creator(), 1)
    shutil.rmtree(path.parents[0])
    plt.show()


def collect_corn_expert_trajectories(path=None):
    action_sequence = np.zeros(53, dtype=int)
    action_sequence[15] = N_ACTIONS - 1
    expert_policy = OpenLoopPolicy(action_sequence)

    env = DummyVecEnv([env_creator])
    if path is None:
        path = PROJECT_PATH.joinpath('expert_trajectories',
                                          'corn_expert_trajectories.pkl')
    collect_expert_trajectories(env, path=path, expert_policy=expert_policy,
                                rollout_episodes=10)


def get_action_distributions(obs, policy):
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs)
    dist = policy.get_distribution(obs)
    return dist.distribution.probs.detach().numpy()


if __name__ == '__main__':
    example()


















def compare_ent_coef_bc(coeffs):
    action_sequence = np.zeros(53, dtype=int)
    action_sequence[15] = N_ACTIONS - 1
    expert_policy = OpenLoopPolicy(action_sequence)

    env = DummyVecEnv([env_creator])
    path = collect_expert_trajectories(env,
                                       path=None,
                                       expert_policy=expert_policy,
                                       rollout_episodes=10)
    for c in coeffs:
        bc_kwargs = dict(ent_weight=c, l2_weight=0)

        env = DummyVecEnv([env_creator])
        model = PPO('MlpPolicy', env, verbose=0, learning_rate=1e-4,
                    n_steps=53, batch_size=128,
                    n_epochs=5, ent_coef=1e-4)

        train_bc(env, path, model.policy, epochs=40, **bc_kwargs)

        observations = np.empty((53, 24))
        dummy_env = env_creator()
        observations[0, :] = dummy_env.reset()
        for i in range(1, 53):
            action = 6 if i == 16 else 0
            observations[i, :], r, done, info = dummy_env.step(action)
        probs = get_action_distributions(observations, model.policy)
        plt.figure()
        plt.plot(probs)
        plt.title(f'Action distribution after BC, ent coef {c}')
        plt.show()


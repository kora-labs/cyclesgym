import wandb
from stable_baselines3 import PPO
from train import Train
from eval import _evaluate_policy
from pathlib import Path
from cyclesgym.paths import PROJECT_PATH
import numpy as np
import matplotlib.pyplot as plt
from cyclesgym.envs.implementers import RotationPlanter


def plot_crop_planning_policy_prob(episode_probs):
    from matplotlib.cm import get_cmap
    color = get_cmap('Accent').colors

    crop_prob = np.array(list(zip(*episode_probs))[0]).squeeze()
    planting_prob = np.array(list(zip(*episode_probs))[1]).squeeze()
    plt.stackplot(1 + np.arange(crop_prob.shape[0]), *crop_prob.T)
    plt.xticks(1 + np.arange(crop_prob.shape[0]))
    plt.xlabel('Years')
    plt.ylabel('Probability')
    plt.xlim(1, crop_prob.shape[0])
    plt.ylim(0, 1)
    plt.show()

    plt.stackplot(1 + np.arange(planting_prob.shape[0]), *planting_prob.T,
                  colors=color[:planting_prob.shape[0]])
    #plt.colorbar()
    plt.xticks(1 + np.arange(planting_prob.shape[0]))
    plt.xlabel('Years')
    plt.ylabel('Probability')
    plt.xlim(1, planting_prob.shape[0])
    plt.ylim(0, 1)
    plt.show()


def plot_crop_planning_policy_det(episode_actions, env):
    from matplotlib.cm import get_cmap
    color = get_cmap('Accent').colors
    episode_actions = episode_actions.squeeze()
    planting_date = episode_actions[:, 1]
    planting_date = 90 + planting_date * 7
    crop = episode_actions[:, 0]
    plt.bar(1 + np.arange(planting_date.size), planting_date,
            color=[color[c] for c in crop])
    plt.xticks(1 + np.arange(planting_date.size))
    plt.xlabel('Years')
    plt.ylabel('Planting date [DOY]')
    plt.xlim(0.5, planting_date.size+0.5)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    config = dict(total_timesteps=1000000, eval_freq=1000, n_steps=80, batch_size=64, n_epochs=10, run_id=0,
                  norm_reward=True, stats_path='runs/vec_normalize.pkl',
                  method="PPO", verbose=1, n_process=8, device='auto')

    manager = Train(config)

    train_env = manager.env_maker(training=False, start_year=1980, end_year=1998)
    test_env = manager.env_maker(training=False, start_year=1998, end_year=2016)

    file = PROJECT_PATH.joinpath('logs/best_model')
    print(Path.is_file(file))
    model = PPO.load(file, device='cpu')

    mean_reward, std_reward, episode_actions, episode_rewards, episode_probs = \
        _evaluate_policy(model, train_env,  n_eval_episodes=1, deterministic=True)
    print(mean_reward/19.)

    plot_crop_planning_policy_prob(episode_probs[0])
    plot_crop_planning_policy_det(episode_actions, train_env)



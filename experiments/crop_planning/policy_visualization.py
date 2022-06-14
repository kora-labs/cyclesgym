from stable_baselines3 import PPO
from train import Train
from cyclesgym.utils import _evaluate_policy
from cyclesgym.paths import PROJECT_PATH
from cyclesgym.utils.plot_utils import set_up_plt
import numpy as np
import matplotlib.pyplot as plt
import wandb
from matplotlib.cm import get_cmap

set_up_plt()


def plot_crop_planning_policy_prob(episode_probs):
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
    plt.xticks(1 + np.arange(planting_prob.shape[0]))
    plt.xlabel('Years')
    plt.ylabel('Probability')
    plt.xlim(1, planting_prob.shape[0])
    plt.ylim(0, 1)
    plt.show()


def plot_crop_planning_policy_det(episode_actions, episode_rewards, run, eval_env_class, eval_test):
    color = get_cmap('Accent').colors

    if 'Rotation' in eval_env_class:
        title = 'Rotation observations'
    else:
        title = 'SoilN observations'

    episode_actions = episode_actions.squeeze()
    planting_date = episode_actions[:, 1]
    planting_date = 90 + planting_date * 7
    crop = episode_actions[:, 0]
    fig, ax = plt.subplots(2, 1, sharex=True)

    x = 1980 + np.arange(planting_date.size)

    ax[0].bar(x, np.array(episode_rewards)/1000.,
              color=[color[c] for c in crop])
    ax[0].set_ylabel('Year reward [k\$]')
    ax[0].set_title(title)

    ax[1].scatter(x, planting_date, color=[color[c] for c in crop],
                  s=100, marker='o')
    ax[1].set_xticks(x)
    ax[1].set_xlabel('Years')
    ax[1].set_ylabel('Planting date [DOY]')
    ax[1].set_xlim(min(x) + 0.5, max(x) + 0.5)

    fig.autofmt_xdate()

    plt.legend()
    plt.savefig(PROJECT_PATH.joinpath('figures/crop_planning_policies/' + run.path[-1] + '_' +\
                eval_test + '.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    config = dict(train_start_year=1980, train_end_year=1998, eval_start_year=1998, eval_end_year=2016,
                  total_timesteps=1000000, eval_freq=1000, n_steps=80, batch_size=64, n_epochs=10, run_id=0,
                  norm_reward=True, method="PPO", verbose=1, n_process=8, device='auto')
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("koralabs/experiments_crop_planning")

    summary_list, summary_at_max, config_list, name_list = [], [], [], []
    for run in runs:
        dir = [dir for dir in PROJECT_PATH.joinpath('wandb').iterdir() if run.path[-1] in str(dir)]
        if len(dir) > 0:
            print(run)
            dir = dir[0]
            file = dir.joinpath('files/models/eval_det/best_model')
            run_config = run.config
            print(run_config)

            eval_env_class = 'CropPlanningFixedPlanting'
            if run_config.get('non_adaptive', False) == 'True':
                eval_env_class = 'CropPlanningFixedPlantingRotationObserver'

            config['eval_env_class'] = eval_env_class

            envs = Train(config).create_envs()

            model = PPO.load(file, device='cpu')

            def get_plot(model, env, run, eval_env_class, eval_test, deterministic=True):
                mean_reward, std_reward, episode_actions, episode_rewards, episode_probs, episode_action_rewards = \
                    _evaluate_policy(model, env,  n_eval_episodes=1, deterministic=deterministic)
                plot_crop_planning_policy_det(episode_actions, episode_action_rewards, run, eval_env_class, eval_test)

            for env, eval_test in zip(envs, ['train', 'rs98', 'nh98', 'nh15']):
                get_plot(model, env, run, eval_env_class, eval_test)



import wandb
from stable_baselines3 import PPO
from train import Train
from eval import _evaluate_policy
from pathlib import Path
from cyclesgym.paths import PROJECT_PATH

if __name__ == '__main__':
    config = dict(total_timesteps=1000000, eval_freq=1000, n_steps=80, batch_size=64, n_epochs=10, run_id=0,
                  norm_reward=True, stats_path='runs/vec_normalize.pkl',
                  method="PPO", verbose=1, n_process=8, device='auto')

    manager = Train(config)

    train_env = manager.env_maker(training=False, start_year=1980, end_year=2000)
    test_env = manager.env_maker(training=False, start_year=2000, end_year=2016)

    file = PROJECT_PATH.joinpath('logs/best_model')
    print(Path.is_file(file))
    model = PPO.load(file, device='cpu')

    mean_reward, std_reward, episode_actions, episode_rewards = _evaluate_policy(model, train_env,  n_eval_episodes=1,
                                                                           deterministic=False)
    print(mean_reward)
    mean_reward, std_reward, episode_actions, episode_rewards = _evaluate_policy(model, test_env, n_eval_episodes=1,
                                                                                 deterministic=False)
    print(mean_reward)


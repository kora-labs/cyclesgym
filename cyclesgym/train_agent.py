from cyclesgym.env import CornEnv
import time
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy


def main():
    # Create environment
    models_dir = Path.cwd().parent.joinpath('agents', 'corn', 'ppo')
    models_dir.mkdir(exist_ok=True, parents=True)
    env = CornEnv('ContinuousCorn.ctrl')

    for i in range(1):
        env_monitor = Monitor(env, filename=str(models_dir.joinpath(f'{i}')))
        vec_env = SubprocVecEnv([lambda: env] * 8) 

        # Instantiate the agent
        # model = DQN('MlpPolicy', vec_env, verbose=0, learning_rate=1e-3,
        #             learning_starts=0)
        model = PPO('MlpPolicy', vec_env, verbose=0,  learning_rate=1e-3)

        # Save untrained agent
        model.save(models_dir.joinpath('corn_untrained'))

        # Train the agent
        t = time.time()
        model.learn(total_timesteps=int(5e4))
        training_time = time.time() - t

        print(f'Training time: {training_time}')

        model.save(models_dir.joinpath(f'corn_dqn_trained{i}'))

        mean_r, std_r, = evaluate_policy(model, env=DummyVecEnv([lambda: env]),
                                         n_eval_episodes=1)
        print(f'Trained reward {mean_r}')


    # Plot training curve
    x, y = ts2xy(load_results(models_dir), 'episodes')
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()

from cyclesgym.envs.crop_planning import CropPlanning
import time
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import warnings
from cyclesgym.utils.paths import FIGURES_PATH, DATA_PATH

warnings.filterwarnings("ignore")


def time_n_envs(env_fun, n_envs=1, repetitions=1, total_timesteps=int(5e4)):
    def env_creator():
        mask = np.zeros(26, dtype=bool)
        mask[-2:] = True
        base_env = env_fun()
        return Monitor(base_env)
        # return gym.make('CartPole-v1')
    training_times = np.zeros(repetitions, dtype=float)
    for i in range(repetitions):
        vec_env = SubprocVecEnv([env_creator] * n_envs)
        model = PPO('MlpPolicy', vec_env, verbose=0,
                    learning_rate=1e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    ent_coef=1e-4,
                    seed=0,
                    device='cpu')
        t = time.time()
        model.learn(total_timesteps=total_timesteps)
        training_times[i] = time.time() - t
    return training_times


def main():
    n_envs = np.logspace(3, 4, num=2, base=2.0).astype(int)[::-1]
    means = np.zeros(n_envs.size)
    stds = np.zeros(n_envs.size)
    reps = 1
    total_timesteps = int(10)
    for i, n in enumerate(n_envs):
        training_times = time_n_envs(lambda: CropPlanning(start_year=1980, end_year=1982,
                                                          rotation_crops=['CornRM.100', 'SoybeanMG.3']),
                                     n, repetitions=reps,
                                     total_timesteps=total_timesteps)
        print(f'{n_envs} parallel environments for {total_timesteps} time steps'
              f'\nTraining times {training_times}'
              f'\nMean {training_times.mean()} str {training_times.std()}')
        means[i] = training_times.mean()
        stds[i] = training_times.std()
        print(f'Number of envs: {n}\t'
              f'Training time {means[i]} +- {stds[i]}'
              f'\tTime per env {means[i]/n}')

    plt.figure()
    plt.bar(np.arange(n_envs.size), means, yerr=stds, tick_label=[str(i) for i in n_envs])
    data_path = DATA_PATH.joinpath('SubProcEnv_profiling.npz')
    fig_path = FIGURES_PATH.joinpath('SubProcEnv_profiling.pdf')
    plt.savefig(fig_path, format='pdf')
    np.savez(data_path, mean=means, stds=stds, n_envs=n_envs)
    plt.show()


if __name__ == '__main__':
    main()
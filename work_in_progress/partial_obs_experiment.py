from cyclesgym.envs import PartialObsCornEnv
import numpy as np
import time
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from cyclesgym.utils.paths import AGENTS_PATH

def train(mask):
    # Create environment
    models_dir = AGENTS_PATH.joinpath('corn', 'ppo')
    models_dir.mkdir(exist_ok=True, parents=True)
    env = PartialObsCornEnv('ContinuousCorn.ctrl', mask=mask)

    vec_env = SubprocVecEnv([lambda: PartialObsCornEnv('ContinuousCorn.ctrl',
                                                       mask=mask)] * 4)

    # Instantiate the agent
    model = PPO('MlpPolicy', vec_env, verbose=0,
                learning_rate=1e-4,
                n_steps=53,
                batch_size=128,
                n_epochs=5,
                gae_lambda=0.95,
                ent_coef=1e-4,
                seed=0,
                policy_kwargs={'net_arch': [8, {'vf': [8]}]})

    # Save untrained agent
    # model.save(models_dir.joinpath('corn_untrained'))

    # Train the agent
    t = time.time()
    model.learn(total_timesteps=int(53 * 100))
    training_time = time.time() - t

    print(f'Training time: {training_time}')

    # model.save(models_dir.joinpath(f'corn_dqn_trained{config["seed"]}'))

    mean_r, std_r, = evaluate_policy(model, env=DummyVecEnv([lambda: env]),
                                     n_eval_episodes=1)

    # Cost array
    c = np.full(24, 0.1, dtype=float)
    obs_cost = np.dot(c, mask)

    print(f'Trained reward {mean_r}, observation cost: {obs_cost}')
    return mean_r - obs_cost


if __name__ == '__main__':
    r = train(mask=np.random.choice(2, 24).astype(bool))


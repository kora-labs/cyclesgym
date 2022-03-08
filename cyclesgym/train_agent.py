from cyclesgym.envs.common import PartialObsCornEnv
import time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback

import yaml
from yaml import Loader
import multiprocessing

from cyclesgym.pretrain_utils import train_bc, collect_corn_expert_trajectories
from cyclesgym.paths import PROJECT_PATH


def create_net_arch(config):
    net_arch = []
    if config['common_layers'] > 0:
        net_arch.extend([config['common_neurons']] * config['common_layers'])

    # In case there is no common layer and no individual layer for vf or pi
    if not net_arch and (config['pi_layers'] == 0 or config['vf_layers'] > 0):
        net_arch = [16]

    d = {}
    if config['pi_layers'] > 0:
        d.update({'pi': [config['pi_neurons']] * config['pi_layers']})
    if config['vf_layers'] > 0:
        d.update({'vf': [config['vf_neurons']] * config['vf_layers']})
    if d:
        net_arch.append(d)
    return net_arch


def train():
    run = wandb.init(sync_tensorboard=True) # auto-upload sb3's tensorboard metrics
    config = wandb.config
    set_random_seed(config['seed'])

    # Create environment
    def env_creator():
        mask = np.zeros(26, dtype=bool)
        mask[-2:] = True
        return Monitor(PartialObsCornEnv('ContinuousCorn',
                                         delta=7, maxN=150, n_actions=11,
                                         mask=mask))
        # return gym.make('CartPole-v1')

    # TODO: time vec env benefits for large simulations
    vec_env = SubprocVecEnv([env_creator] * 4)

    # Instantiate the agent
    net_arch = create_net_arch(config)
    model = PPO('MlpPolicy', vec_env, verbose=0,
                tensorboard_log=f'./runs/{run.id}',  # TODO: Change log dir
                learning_rate=config['learning_rate'],
                n_steps=config['n_steps'],
                batch_size=config['batch_size'],
                n_epochs=config['n_epochs'],
                gae_lambda=config['gae_lambda'],
                ent_coef=config['ent_coef'],
                seed=config['seed'],
                policy_kwargs={'net_arch': net_arch})

    if config['bc']:
        path = PROJECT_PATH.joinpath('expert_trajectories',
                                              'corn_expert_trajectories.pkl')
        if not path.is_file():
            collect_corn_expert_trajectories(path)

        bc_kwargs = dict(ent_weight=1e-1, l2_weight=0.0)
        train_bc(env, path, model.policy, epochs=40, **bc_kwargs)

        # TODO: get kwargs from PPO and make sure optimizer state is new

        det_r_bc, _ = evaluate_policy(model,
                                            env=DummyVecEnv([lambda: env]),
                                            n_eval_episodes=1,
                                            deterministic=True)
        stoc_r_bc, _ = evaluate_policy(model,
                                            env=DummyVecEnv([lambda: env]),
                                            n_eval_episodes=10,
                                            deterministic=False)
        print(
            f'Reward after behavioral cloning: deterministic {det_r_bc}\t'
            f'stochastic {stoc_r_bc}')
    else:
        det_r_bc = None
        stoc_r_bc = None

    # Train the agent
    t = time.time()
    model.learn(total_timesteps=int(53 * config['n_years']),
                callback=WandbCallback(gradient_save_freq=0,  # Don't save gradients
                                       verbose=2)
                )
    training_time = time.time() - t

    print(f'Training time: {training_time}')

    mean_r_det, _ = evaluate_policy(model,
                                    env=DummyVecEnv([env_creator]),
                                    n_eval_episodes=1,
                                    deterministic=True)
    mean_r_stoc, std_r_stoc = evaluate_policy(model,
                                              env=DummyVecEnv([env_creator]),
                                              n_eval_episodes=5,
                                              deterministic=False)

    wandb.log({'deterministic_return': mean_r_det,
               'stochastic_return_mean': mean_r_stoc,
               'stochastic_return_std': std_r_stoc,
               'training_time': training_time})
               # 'det_r_bc': det_r_bc, 'stoc_r_bc': stoc_r_bc})
    print(f'Trained reward {mean_r_det}')
    run.finish()


if __name__ == '__main__':
    with open('ppo_sweep_config.yaml', 'r') as fp:
        sweep_config = yaml.load(fp, Loader=Loader)
    print(sweep_config)
    sweep_id = wandb.sweep(sweep_config,
                           entity='matteotu',
                           project='sweep_cart_pole_with_monitor')
    t = time.time()
    procs = []
    for i in range(4):
        p = multiprocessing.Process(target=wandb.agent,
                                    args=(sweep_id,),
                                    kwargs={'function': train}
                                            #'count': 10}
                                    )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


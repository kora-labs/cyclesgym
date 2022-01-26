from cyclesgym.env import CornEnv
import time
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
import yaml
from yaml import Loader
import multiprocessing


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
    wandb.init()
    config = wandb.config

    # Create environmentFix
    models_dir = Path.cwd().parent.joinpath('agents', 'corn', 'ppo')
    models_dir.mkdir(exist_ok=True, parents=True)
    env = CornEnv('ContinuousCorn.ctrl')

    vec_env = SubprocVecEnv([lambda: env] * 2)

    # Instantiate the agent
    net_arch = create_net_arch(config)
    model = PPO('MlpPolicy', vec_env, verbose=0,
                learning_rate=config['learning_rate'],
                n_steps=config['n_steps'],
                batch_size=config['batch_size'],
                n_epochs=config['n_epochs'],
                gae_lambda=config['gae_lambda'],
                ent_coef=config['ent_coef'],
                seed=config['seed'],
                policy_kwargs={'net_arch': net_arch})

    # Save untrained agent
    model.save(models_dir.joinpath('corn_untrained'))

    # Train the agent
    t = time.time()
    model.learn(total_timesteps=int(53 * config['n_years']))
    training_time = time.time() - t

    print(f'Training time: {training_time}')

    # model.save(models_dir.joinpath(f'corn_dqn_trained{config["seed"]}'))

    mean_r, std_r, = evaluate_policy(model, env=DummyVecEnv([lambda: env]),
                                     n_eval_episodes=1)
    wandb.log({'return': mean_r, 'training_time': training_time})
    print(f'Trained reward {mean_r}')


if __name__ == '__main__':
    with open('ppo_sweep_config.yaml', 'r') as fp:
        sweep_config = yaml.load(fp, Loader=Loader)
    print(sweep_config)
    sweep_id = wandb.sweep(sweep_config,
                           entity='matteotu',
                           project='ppo_sweep')
    t = time.time()
    procs = []
    for i in range(8):
        p = multiprocessing.Process(target=wandb.agent,
                                    args=(sweep_id,),
                                    kwargs={'function': train}
                                            #'count': 10}
                                    )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


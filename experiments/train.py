import time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
# from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from eval import _evaluate_policy
import gym
from cyclesgym.envs.corn import CornShuffledWeather
from corn_soil_refined import CornSoilRefined
import wandb
from wandb.integration.sb3 import WandbCallback
import sys

from cyclesgym.dummy_policies import OpenLoopPolicy
import expert


class Train:
    """ Trainer object to wrap model training and handle environment creation, evaluation """

    def __init__(self, experiment_config) -> None:
        self.config = experiment_config
        # rl config is configured from wandb config


    def env_maker(self, training = True, n_procs = 1, soil_env = False, start_year = 1980, end_year = 1980,
        sampling_start_year=1980, sampling_end_year=2013,
        n_weather_samples=100, fixed_weather = True):
        if not training:
            n_procs = 1

        def make_env():
            # creates a function returning the basic env. Used by SubprocVecEnv later to create a
            # vectorized environment
            def _f():
                if soil_env:
                    env = CornSoilRefined(delta=7, maxN=150, n_actions=self.config['n_actions'],
                        start_year = start_year, end_year = end_year,
                        sampling_start_year=sampling_start_year,
                        sampling_end_year=sampling_end_year,
                        n_weather_samples=n_weather_samples,
                        fixed_weather=fixed_weather)
                else:
                    if fixed_weather:
                        env = CornShuffledWeather(delta=7, maxN=150, n_actions=self.config['n_actions'],
                            start_year = start_year, end_year = end_year,
                            sampling_start_year=sampling_start_year,
                            sampling_end_year=sampling_end_year,
                            n_weather_samples=n_weather_samples)
                    else:
                        env = Corn(delta=7, maxN=150, n_actions=self.config['n_actions'],
                            start_year = start_year, end_year = end_year)

                #env = Monitor(env, 'runs')
                env = gym.wrappers.RecordEpisodeStatistics(env)
                return env
            return _f

        env = SubprocVecEnv([make_env() for i in range(n_procs)], start_method='fork')

        env = VecMonitor(env, 'runs')

        #only norm the reward if we selected to do so and if we are in training
        norm_reward = (training and self.config['norm_reward'])

        #high clipping values so that they effectively get ignored
        env = VecNormalize(env, norm_obs=True, norm_reward= norm_reward, clip_obs=20000., clip_reward=20000.)

        return env

    def train(self):
        
        train_env = self.env_maker(training = True, n_procs=1, soil_env = self.config['soil_env'],
         start_year = self.config['start_year'], end_year = self.config['end_year'], 
         sampling_start_year=self.config['sampling_start_year'],
         sampling_end_year=self.config['sampling_end_year'],
         n_weather_samples=self.config['n_weather_samples'],
         fixed_weather = self.config['fixed_weather'])
        if config["method"] == "A2C":
            model = A2C('MlpPolicy', train_env, verbose=0, tensorboard_log=f"runs")
        elif config["method"] == "PPO":
            model = PPO('MlpPolicy', train_env, verbose=0, tensorboard_log=f"runs")
        elif config["method"] == "DQN":
            model = DQN('MlpPolicy', train_env, verbose=0, tensorboard_log=f"runs")
        else:
            raise Exception("Not an RL method that has been implemented")

        """
        wandb_callback = WandbCallback(gradient_save_freq=0,  # 100,  # Don't save gradients
                                       verbose=2,
                                       model_save_path=WANDB_PATH.joinpath(
                                           f'./runs/{run.id}'),
                                       model_save_freq=20, )
        
        eval_freq = self.config['eval_freq']
        eval_callback_sto = EvalCallback_(self.env_maker(), eval_freq=eval_freq, deterministic=False,
                                      n_eval_episodes=10)
        eval_callback_det = EvalCallback_(self.env_maker(), eval_freq=eval_freq, deterministic=True,
                                      n_eval_episodes=1)
        callbacks = [eval_callback, wandb_callback]
        """
        # The test environment will automatically have the same observation normalization applied to it by 
        # EvalCallBack
        eval_env = self.env_maker(training = False, soil_env = self.config['soil_env'],
            start_year = self.config['start_year'], end_year = self.config['end_year'],
            sampling_start_year=self.config['sampling_start_year'],
            sampling_end_year=self.config['sampling_end_year'],
            n_weather_samples=self.config['n_weather_samples'],
            fixed_weather = self.config['fixed_weather'])
        eval_env.training = False
        eval_env.norm_reward = False
        eval_callback_det = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='runs', eval_freq=config['eval_freq'],
                             deterministic=True, render=False)
        eval_callback_sto = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='runs', eval_freq=config['eval_freq'],
                             deterministic=False, render=False)
        callback = [WandbCallback(), eval_callback_det, eval_callback_sto]
        model.learn(total_timesteps=self.config["total_timesteps"], callback=callback)
        model.save(str(self.config['run_id'])+'.zip')
        train_env.save(self.config['stats_path'])
        return model, eval_env

    def evaluate_log(self, model, eval_env):
        """
        Runs policy deterministically (1 episode) and stochastically (5 episodes)
        logs the fertilization actions taken by the model

        Parameters
        ----------
        model: trained agent
        eval_env

        Returns
        -------
        mean deterministic reward

        """
        #list, list, numpy array, list
        mean_r_det, _, actions_det, episode_rewards_det = _evaluate_policy(model,
                                                                           env=eval_env,
                                                                           n_eval_episodes=1,
                                                                           deterministic=True)

        mean_r_stoc, std_r_stoc, actions_stoc, episode_rewards_stoc = _evaluate_policy(model,
                                                                                       env=eval_env,
                                                                                       n_eval_episodes=5,
                                                                                       deterministic=False)
    
        T = actions_stoc.shape[1]
        wandb.log({'deterministic_return': mean_r_det,
                   'stochastic_return_mean': mean_r_stoc,
                   'stochastic_return_std': std_r_stoc,
                   })
        episode_actions_names = [*list(f"det{i + 1}" for i in range(len(actions_det))),
                                 *list(f"stoc{i + 1}" for i in range(len(actions_stoc)))]
        episode_actions = [*actions_det, *actions_stoc]
        fertilizer_table = wandb.Table(
            columns=['Run', 'Total Fertilizer', *[f'Week{i}' for i in range(T)]])
        for i in range(len(episode_actions)):
            acts = episode_actions[i]
            data = [[week, fert] for (week, fert) in zip(range(T), acts)]
            table = wandb.Table(data=data, columns=['Week', 'N added'])
            fertilizer_table.add_data(
                *[episode_actions_names[i], np.sum(acts), *acts])
            wandb.log({f'train/actions/{episode_actions_names[i]}': wandb.plot.bar(table, 'Week', 'N added',
                                                                                   title=f'Training action sequence {episode_actions_names[i]}')})
        wandb.log({'train/fertilizer': fertilizer_table})

        ## create a plot of the reward in each year
        ## create a plot of fertilizer cost in each year
        return mean_r_det

    def eval_experts(self, action_series, eval_env, name):
        action_series_int = np.array(action_series, dtype=int)
        expert_policy = OpenLoopPolicy(action_series_int)
        r, _, _, _ = _evaluate_policy(expert_policy,
                                eval_env,
                                n_eval_episodes=10,
                                deterministic=True)
        wandb.log({f'train/baseline/'+name: r})
        #wandb.log({f'train/baseline/'+name+'_rseq': r_seq})

def make_multi_year(action_seq, n):
    # n is an int for number of years
    return np.concatenate([action_seq, np.repeat(action_seq[1:], n)])


if __name__ == '__main__':

    if(len(sys.argv) > 1):
        method = str(sys.argv[1])
    else:
        method = "A2C"

    config = dict(total_timesteps = 500000, eval_freq = 1000, run_id = 0,
                norm_reward = True,  stats_path = 'runs/vec_normalize.pkl',
                method = "PPO", n_actions = 11, soil_env=True, start_year = 1980,
                end_year = 1981, sampling_start_year=1980, sampling_end_year=2013,
                n_weather_samples=100, fixed_weather = True)

    wandb.init(
    config=config,
    sync_tensorboard=True,
    project="agro-rl",
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=True,
    )

    config = wandb.config
    
    trainer = Train(config)
    model, eval_env = trainer.train()
    # Load the saved statistics

    #eval_env = VecNormalize.load(config['stats_path'], eval_env)
    #  do not update moving averages at test time
    eval_env.training = False
    # reward normalization is not needed at test time
    eval_env.norm_reward = False
    trainer.evaluate_log(model, eval_env)
    
    agro_exact_sequence = expert.create_action_sequence(doy=[110, 155], weight=[35, 120],
                                         maxN=150,
                                         n_actions=config['n_actions'],
                                         delta_t=7)

    nonsense_exact_sequence = expert.create_action_sequence(doy=[110, 155, 300], weight=[35, 120, 50],
                                         maxN=150,
                                         n_actions=config['n_actions'],
                                         delta_t=7)

    cycles_exact_sequence = expert.create_action_sequence(doy=110, weight=150,
                                         maxN=150,
                                         n_actions=config['n_actions'],
                                         delta_t=7)

    organic_exact_sequence = expert.create_action_sequence(doy=110, weight=0,
                                         maxN=150,
                                         n_actions=config['n_actions'],
                                         delta_t=7)
    
    n = config['end_year'] - config['start_year']
    agro_exact_sequence = make_multi_year(agro_exact_sequence, n)
    nonsense_exact_sequence = make_multi_year(nonsense_exact_sequence, n)
    cycles_exact_sequence = make_multi_year(cycles_exact_sequence, n)
    organic_exact_sequence = make_multi_year(organic_exact_sequence, n)

    trainer.eval_experts(organic_exact_sequence, eval_env, "organic")
    trainer.eval_experts(agro_exact_sequence, eval_env, "agro")
    trainer.eval_experts(cycles_exact_sequence, eval_env, "cycles")
    
    trainer.eval_experts(nonsense_exact_sequence, eval_env, "nonsense")
    


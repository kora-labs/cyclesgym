import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from cyclesgym.envs.corn import Corn
from cyclesgym.utils.utils import EvalCallbackCustom, _evaluate_policy
from cyclesgym.utils.wandb_utils import WANDB_ENTITY, FERTILIZATION_EXPERIMENT
import gym
from corn_soil_refined import CornSoilRefined, NonAdaptiveCorn
import wandb
from wandb.integration.sb3 import WandbCallback
from pathlib import Path
from cyclesgym.utils.paths import PROJECT_PATH, CYCLES_PATH
from cyclesgym.envs.weather_generator import WeatherShuffler
import sys


from cyclesgym.policies.dummy_policies import OpenLoopPolicy
import expert
import argparse
import random


class Train:
    """ Trainer object to wrap model training and handle environment creation, evaluation """

    def __init__(self, experiment_config, with_obs_year) -> None:
        self.config = experiment_config
        self.with_obs_year = with_obs_year
        self.dir = wandb.run.dir
        self.model_dir = Path(self.dir).joinpath('models')
        # rl config is configured from wandb config

    def env_maker(self, training = True, n_procs = 1, soil_env = False, start_year = 1980, end_year = 1980,
        sampling_start_year=1980, sampling_end_year=2013,
        n_weather_samples=100, fixed_weather = True, with_obs_year=False,
        nonadaptive=False, new_holland=False):

        def make_env():
            # creates a function returning the basic env. Used by SubprocVecEnv later to create a
            # vectorized environment
            def _f():
                if nonadaptive:
                    env = NonAdaptiveCorn(delta=7, maxN=150, n_actions=self.config['n_actions'],
                            start_year = start_year, end_year = end_year,
                            sampling_start_year=sampling_start_year,
                            sampling_end_year=sampling_end_year,
                            n_weather_samples=n_weather_samples,
                            fixed_weather=fixed_weather,
                            with_obs_year=with_obs_year,
                            new_holland=new_holland)
                else:
                    if soil_env:
                        env = CornSoilRefined(delta=7, maxN=150, n_actions=self.config['n_actions'],
                            start_year = start_year, end_year = end_year,
                            sampling_start_year=sampling_start_year,
                            sampling_end_year=sampling_end_year,
                            n_weather_samples=n_weather_samples,
                            fixed_weather=fixed_weather,
                            with_obs_year=with_obs_year,
                            new_holland=new_holland)
                    else:
                        if fixed_weather:
                            env = Corn(delta=7, maxN=150, n_actions=self.config['n_actions'],
                                start_year = start_year, end_year = end_year)
                        else:
                            target_year_range = np.arange(start_year, end_year + 1)
                            weather_generator_kwargs = dict(
                                n_weather_samples=n_weather_samples,
                                sampling_start_year=sampling_start_year,
                                sampling_end_year=sampling_end_year,
                                target_year_range=target_year_range,
                                base_weather_file=CYCLES_PATH.joinpath('input', 'RockSprings.weather'))
                            env = Corn(delta=7, maxN=150, n_actions=self.config['n_actions'],
                                       start_year=start_year, end_year=end_year,
                                       weather_generator_class=WeatherShuffler,
                                       weather_generator_kwargs=weather_generator_kwargs)

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

    def long_env_maker(self, training = True, n_procs = 1, soil_env = False, start_year = 1980, end_year = 1980,
        sampling_start_year=1980, sampling_end_year=2013,
        n_weather_samples=100, fixed_weather = True, with_obs_year=False, nonadaptive=False):
        """
        for single year testing we want to have an env that is identical to others but just a longer time horizon
        """
        def f(years):
            return self.env_maker(training = False, soil_env = self.config['soil_env'],
                        start_year = self.config['start_year'], end_year = self.config['end_year']+years-1,
                        sampling_start_year=self.config['sampling_start_year'],
                        sampling_end_year=self.config['sampling_end_year'],
                        n_weather_samples=self.config['n_weather_samples'],
                        fixed_weather = self.config['fixed_weather'],
                        with_obs_year=self.with_obs_year,
                        nonadaptive=self.config['nonadaptive'])
        return f

    def get_envs(self, n_procs, new_holland=False, plus_horizon=0):
        """
        Returns some environments given n_procs. Used because I often want the same settings
        but a different n_procs for policy visualization and baseline evaluations
        """
        hold_out_sampling_start_year = self.config['sampling_end_year'] + 1
        assert (hold_out_sampling_start_year <= 2015)
        hold_out_sampling_end_year = 2015 #just last year in the weather data
        duration = self.config['end_year'] - self.config['start_year'] 

        # The test environment will automatically have the same observation normalization applied to it by 
        # EvalCallBack
        eval_env_train = self.env_maker(training = False, n_procs=n_procs,
            soil_env = self.config['soil_env'],
            start_year = self.config['start_year'], end_year = self.config['end_year']+plus_horizon,
            sampling_start_year=self.config['sampling_start_year'],
            sampling_end_year=self.config['sampling_end_year'],
            n_weather_samples=self.config['n_weather_samples'],
            fixed_weather = self.config['fixed_weather'],
            with_obs_year=self.with_obs_year,
            nonadaptive=self.config['nonadaptive'],
            new_holland=new_holland)

        #the out of sample weather env
        start_year = hold_out_sampling_start_year
        end_year = hold_out_sampling_start_year+duration
        if self.config['fixed_weather']:
            end_year = self.config['sampling_end_year']
            start_year = end_year-duration
        eval_env_test = self.env_maker(training = False, n_procs=n_procs,
            soil_env = self.config['soil_env'],
            start_year = start_year, end_year = end_year+plus_horizon,
            sampling_start_year=hold_out_sampling_start_year,
            sampling_end_year=hold_out_sampling_end_year,
            n_weather_samples=self.config['n_weather_samples'],
            fixed_weather = False,
            with_obs_year=self.with_obs_year,
            nonadaptive=self.config['nonadaptive'],
            new_holland=new_holland)

        eval_env_train.training = False
        eval_env_train.norm_reward = False
        eval_env_test.training = False
        eval_env_test.norm_reward = False

        return eval_env_train, eval_env_test


    def get_eval_callbacks(self):
        """
        generates all callbacks plus test and train envs
        """
        eval_freq = int(self.config['eval_freq'] / self.config['n_process'])
        eval_env_train, eval_env_test = self.get_envs(n_procs=self.config['n_process'])

        eval_callback_test_det = EvalCallbackCustom(eval_env_test, best_model_save_path=None,
            log_path=str(self.model_dir.joinpath('eval_test_det')),
            eval_freq=eval_freq, deterministic=True, render=False,
            eval_prefix='eval_test_det')
        eval_callback_test_sto = EvalCallbackCustom(eval_env_test, best_model_save_path=None,
            log_path=str(self.model_dir.joinpath('eval_test_sto')),
            eval_freq=eval_freq, deterministic=False, render=False,
            eval_prefix='eval_test_sto')

        eval_callback_det = EvalCallbackCustom(eval_env_train, best_model_save_path=None,
            log_path=str(self.model_dir.joinpath('eval_test_det')),
            eval_freq=eval_freq, deterministic=True, render=False,
            eval_prefix='eval_train_det')
        eval_callback_sto = EvalCallbackCustom(eval_env_train, best_model_save_path=str(self.model_dir.joinpath('train_sto')),
            log_path=str(self.model_dir.joinpath('eval_test_det')),
            eval_freq=eval_freq, deterministic=False, render=False,
            eval_prefix='eval_train_sto')

        callback = [WandbCallback(model_save_path=str(self.model_dir), model_save_freq=int(self.config['eval_freq'] / self.config['n_process'])),
            eval_callback_det, eval_callback_sto,
            eval_callback_test_det, eval_callback_test_sto]
        return callback

    def train(self):
        
        train_env = self.env_maker(training = True, n_procs=self.config['n_process'], soil_env = self.config['soil_env'],
         start_year = self.config['start_year'], end_year = self.config['end_year'], 
         sampling_start_year=self.config['sampling_start_year'],
         sampling_end_year=self.config['sampling_end_year'],
         n_weather_samples=self.config['n_weather_samples'],
         fixed_weather = self.config['fixed_weather'],
         with_obs_year=self.with_obs_year,
         nonadaptive=self.config['nonadaptive'])

        train_env.seed(self.config['seed'])

        eval_freq = int(self.config['eval_freq'] / self.config['n_process'])
        duration = self.config['end_year'] - self.config['start_year']
        total_timesteps = self.config["total_years"] * 53
        n_steps = int(self.config['n_steps'] / self.config['n_process'])

        if config["method"] == "A2C":
            model = A2C('MlpPolicy', train_env, verbose=0, tensorboard_log=self.dir)
        elif config["method"] == "PPO":
            model = PPO('MlpPolicy', train_env, verbose=0, n_steps= n_steps, tensorboard_log=self.dir)
        elif config["method"] == "DQN":
            model = DQN('MlpPolicy', train_env, verbose=0, tensorboard_log=self.dir)
        else:
            raise Exception("Not an RL method that has been implemented")


        callback = self.get_eval_callbacks()
        
        model.learn(total_timesteps=total_timesteps, callback=callback, eval_freq=eval_freq)
        model.save(str(self.config['run_id'])+'.zip')
        train_env.save(self.config['stats_path'])
        return model

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

    def eval_openloop(self, action_series, eval_env, name):
        action_series_int = np.array(action_series, dtype=int)
        expert_policy = OpenLoopPolicy(action_series_int)
        r, _ = evaluate_policy(expert_policy,
                                eval_env,
                                n_eval_episodes=100,
                                deterministic=True)
        wandb.log({f'train/baseline/'+name: r})
        return

    def one_year_eval(self, model):
        """
        An evaluation to test the one year policy on 2,5,10 years
        """
        for long_len in [2, 5]:
            #env = long_env(long_len)
            env, _  = trainer.get_envs(n_procs = 1, plus_horizon=long_len-1)
            env = VecNormalize.load(self.config['stats_path'], env)
            env.training = False
            env.norm_reward = False
            
            r_det, _ = evaluate_policy(model,
                                env,
                                n_eval_episodes=20,
                                deterministic=True)
            
            r_sto, _ = evaluate_policy(model,
                                env,
                                n_eval_episodes=20,
                                deterministic=False)
            
            name = "long_eval_det"+str(long_len)
            wandb.log({f'eval/'+name: r_det})
            name = "long_eval_sto"+str(long_len)
            wandb.log({f'eval/'+name: r_sto})
        return

    def eval_baselines(self):
        ## evaluate baseline strategies on the train and test envs
        #make an env on 1 process for open loop policies and vis
        eval_env_train, eval_env_test = self.get_envs(n_procs = 1)

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
        trainer.eval_openloop(organic_exact_sequence, eval_env_train, "organic-train")
        trainer.eval_openloop(agro_exact_sequence, eval_env_train, "agro-train")
        trainer.eval_openloop(cycles_exact_sequence, eval_env_train, "cycles-train")
        trainer.eval_openloop(nonsense_exact_sequence, eval_env_train, "nonsense-train")

        trainer.eval_openloop(organic_exact_sequence, eval_env_test, "organic-test")
        trainer.eval_openloop(agro_exact_sequence, eval_env_test, "agro-test")
        trainer.eval_openloop(cycles_exact_sequence, eval_env_test, "cycles-test")
        trainer.eval_openloop(nonsense_exact_sequence, eval_env_test, "nonsense-test")

        nh_env = self.env_maker(training = False, n_procs=1,
            soil_env = self.config['soil_env'],
            start_year = self.config['start_year'], end_year = self.config['end_year'],
            sampling_start_year=self.config['sampling_start_year'],
            sampling_end_year=self.config['sampling_end_year'],
            n_weather_samples=self.config['n_weather_samples'],
            fixed_weather = self.config['fixed_weather'],
            with_obs_year=self.with_obs_year,
            nonadaptive=self.config['nonadaptive'],
            new_holland=True)

        trainer.eval_openloop(organic_exact_sequence, nh_env, "organic-NH")
        trainer.eval_openloop(agro_exact_sequence, nh_env, "agro-NH")
        trainer.eval_openloop(cycles_exact_sequence, nh_env, "cycles-NH")
        trainer.eval_openloop(nonsense_exact_sequence, nh_env, "nonsense-NH")

        return

    def eval_nh(self,model):
        """
        evaluate the model on new holland training years
        """
        nh_env, _  = trainer.get_envs(n_procs = 1, new_holland=True)
        nh_env = VecNormalize.load(self.config['stats_path'], nh_env)
        nh_env.training = False
        nh_env.norm_reward = False
        
        mean_r, _  = evaluate_policy(model,
           env=nh_env,
           n_eval_episodes=20,
           deterministic=False)
        wandb.log({'NH_return': mean_r})
        return


def make_multi_year(action_seq, n):
    # n is an int for number of years
    return np.concatenate([action_seq, np.tile(action_seq[1:], n)])


if __name__ == '__main__':
    RANDOM_SEED = 0
    """
    torch.manual_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    """
    config = dict(total_years = 3000, eval_freq = 1000, run_id = 0,
                norm_reward = True,
                method = "PPO", n_actions = 11, soil_env=True, start_year = 1980,
                sampling_start_year=1980, sampling_end_year=2005,
                n_weather_samples=100, 
                n_steps = 2048, with_obs_year = True)
    wandb.init(
    config=config,
    sync_tensorboard=True,
    project=FERTILIZATION_EXPERIMENT,
    entity=WANDB_ENTITY,
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=True,
    group="group_name",
    dir=PROJECT_PATH
    )

    

    parser = argparse.ArgumentParser()
    parser.add_argument('-np', '--n-process', type=int, default=1, metavar='N',
                         help='input number of processes for training (default: 1)')
    parser.add_argument('-ey', '--end-year', type=int, default=1980, metavar='N',
                         help='The final year of simulation (default: 1980)')
    parser.add_argument('-na','--nonadaptive', default=False, action='store_true',
        help='Whether to learn a nonadaptive policy')
    parser.add_argument('-fw','--fixed-weather', default=False, action='store_true',
        help='Whether to use a fixed weather')
    parser.add_argument('-s', '--seed', type=int, default=0, metavar='N',
                         help='The random seed used for all number generators')
    parser.add_argument('-b','--baseline', default=False, action='store_true',
        help='Use to only run the baselines')

    parser.add_argument('-p','--posthoc', default=False, action='store_true',
        help='Parse to read in a set of weights that are evaluated')

    args = parser.parse_args()

    wandb.config.update(args)

    
    if wandb.config['posthoc']:
        stats_path = 'data/vec_norms/vec_normalize_1xw45c9p.pkl'
    else:
        stats_path = 'runs/vec_normalize_' + str(wandb.run.id) + '.pkl' 
    
    wandb.config.update({'stats_path': stats_path})

    config = wandb.config

    set_random_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    print("status")
    #if we do a 1 year experiment, we don't include obs year (not useful)
    with_obs_year = config['with_obs_year'] and (config['start_year'] != config['end_year'])
    trainer = Train(config, with_obs_year)
    
    #if trying to get baselines...
    if config['baseline']:
        trainer.eval_baselines()
    else:
        if config['posthoc']:
            file = PROJECT_PATH.joinpath('experiments/data/model.zip')
            model = PPO.load(file, device='cpu')
        else:
            model = trainer.train()

        # Load the saved statistics

        _, eval_env_test = trainer.get_envs(n_procs = 1)

        eval_env_test = VecNormalize.load(config['stats_path'], eval_env_test)
        #  do not update moving averages at test time
        eval_env_test.training = False
        # reward normalization is not needed at test time
        eval_env_test.norm_reward = False
        
        trainer.evaluate_log(model, eval_env_test)
        
        if config['start_year'] == config['end_year']:
            trainer.one_year_eval(model)
        
        trainer.eval_nh(model)






    
    


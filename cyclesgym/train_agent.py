from cyclesgym.env import CornEnv
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy


# Create environment
models_dir = Path.cwd().parent.joinpath('agents')
models_dir.mkdir(exist_ok=True)
env = Monitor(CornEnv('ContinuousCorn.ctrl'),
              filename=str(models_dir.joinpath('dqn')))

# Instantiate the agent
model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-3,
            learning_starts=0)

# Save untrained agent
model.save(models_dir.joinpath('corn_untrained'))


# Train the agent
t = time.time()
model.learn(total_timesteps=int(10e3))
training_time = time.time() - t

print(f'Training time: {training_time}')

model.save(models_dir.joinpath('corn_dqn_trained'))

mean_r, std_r, = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
print(f'Trained reward {mean_r}')


# Plot training curve
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


x, y = ts2xy(load_results(models_dir), 'episodes')
plt.plot(x, smooth(y, 10))
plt.show()

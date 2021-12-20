from cyclesgym.env import CornEnv
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import DQN
import pandas as pd
import json
from scipy.signal import savgol_filter

from stable_baselines3.common.monitor import Monitor, get_monitor_files, \
    LoadMonitorResultsError
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt


def my_load_results(path: Path, window_size: int=5) -> pd.DataFrame:
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv``

    :param path: the directory path containing the log file(s)
    :return: the logged data
    """
    monitor_files = get_monitor_files(str(path))
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *{Monitor.EXT} found in {path}")
    data_frames, headers = [], []
    max_len = 0
    for file_name in monitor_files:
        with open(file_name, "rt") as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            data_frame = pd.read_csv(file_handler, index_col=None)
            headers.append(header)
            data_frame["t"] += header["t_start"]
            max_len = max(max_len, len(data_frame['r']))
        data_frames.append(data_frame)
    rewards = np.full((len(data_frames), max_len), np.nan)
    lengths = np.copy(rewards)
    for i, df in enumerate(data_frames):
        rewards[i, :len(df['r'])] = savgol_filter(df['r'], window_size, 1)
        lengths[i, :len(df['l'])] = df['l']
    data = np.column_stack((np.mean(rewards, axis=0),
                            np.std(rewards, axis=0),
                            np.mean(lengths, axis=0),
                            np.std(lengths, axis=0)))
    columns = ['r_mean', 'r_std', 'l_mean', 'l_std']
    return pd.DataFrame(data=data, columns=columns)


# Create environment
models_dir = Path.cwd().parent.joinpath('agents', 'corn', 'dqn')

df = my_load_results(models_dir)
plt.plot(df['r_mean'])
plt.fill_between(np.arange(len(df['r_mean'])), df['r_mean'] - df['r_std'],
                 df['r_mean'] + df['r_std'], alpha=0.5)
plt.show()

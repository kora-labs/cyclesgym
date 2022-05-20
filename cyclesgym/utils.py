import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
eps = 1e-8


def maximum_absolute_percentage_error(y1, y2):
    return 100*np.max(np.abs(y1-y2)/np.abs(y1+eps))


def mean_absolute_percentage_error(y1, y2):
    return 100*np.mean(np.abs(y1-y2)/np.abs(y1+eps))


def plot_two_environments(df1, df2, labels, columns):
    for col in columns:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        original = df1.iloc[:, col]
        new = df2.iloc[:, col]
        ax1.plot(original, label=labels[0], marker='x')
        ax1.plot(new, label=labels[1], marker='o', mfc='none')
        ax2.plot((100 * (new - original) / (original + 1e-5)), label='Percentage difference', c='green')
        ax1.set_title(df1.columns[col])
        fig.legend()
        plt.show()


def run_env(env, actions_to_use=None):
    i = 0
    observations = []
    actions = []
    while True:
        if actions_to_use is None:
            a = env.action_space.sample()
        else:
            a = actions_to_use[i]

        s, r, done, info = env.step(a)
        observations.append(s)
        actions.append(a)

        if done:
            break
        i = i + 1

    return observations, actions


def compare_env(env1, env2):
    env1.reset()
    env2.reset()

    t = time.time()
    obs_1, actions = run_env(env1)
    time_1 = time.time() - t

    t = time.time()
    obs_2, _ = run_env(env2, actions)
    time_2 = time.time() - t

    obs_1 = np.array(obs_1)
    obs_2 = np.array(obs_2)

    df1 = pd.DataFrame(obs_1)
    df2 = pd.DataFrame(obs_2)
    plot_two_environments(df1, df2, ['1', '2'], range(0, obs_1.shape[1]))

    return obs_1, obs_2, time_1, time_2


def diff_pd(df1, df2):
    """Identify differences between two pandas DataFrames"""
    assert (df1.columns == df2.columns).all(), \
        "DataFrame column names are different"
    if any(df1.dtypes != df2.dtypes):
        "Data Types are different, trying to convert"
        df2 = df2.astype(df1.dtypes)
    if df1.equals(df2):
        return None
    else:
        # need to account for np.nan != np.nan returning True
        diff_mask = (df1 != df2) & ~(df1.isnull() & df2.isnull())
        ne_stacked = diff_mask.stack()
        changed = ne_stacked[ne_stacked]
        changed.index.names = ['id', 'col']
        difference_locations = np.where(diff_mask)
        changed_from = df1.values[difference_locations]
        changed_to = df2.values[difference_locations]
        return pd.DataFrame({'from': changed_from, 'to': changed_to},
                            index=changed.index)
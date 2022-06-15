import pandas as pd
import wandb
import csv
from cyclesgym.utils.paths import PROJECT_PATH

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("koralabs/experiments_crop_planning")

summary_list, summary_at_max, config_list, name_list = [], [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    hist = run.history()
    indx = hist.idxmax(axis=0, skipna=True)['eval_det/mean_reward']
    summary_at_max_train = hist.iloc[indx].to_dict()
    summary_at_max_train = {'max_' + str(key): val for key, val in summary_at_max_train.items()}
    summary_at_max.append(summary_at_max_train)
    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.concat([pd.DataFrame(summary_at_max), pd.DataFrame(summary_list), pd.DataFrame(config_list),
                     pd.DataFrame(name_list, columns=['run_name'])], axis=1)

columns = ['max_eval_det/mean_reward', 'max_eval_det_new_years/mean_reward',
           'max_eval_det_other_loc/mean_reward', 'max_eval_det_other_loc_long/mean_reward']

for c,l in zip(columns, [19., 19., 19., 35]):
    runs_df[c] = runs_df[c] / l


groupby = runs_df.groupby(['fixed_weather', 'env_class'])

results_mean = groupby.mean()[columns]
results_std = groupby.std()[columns]
results_min = groupby.min()[columns]
results_max = groupby.max()[columns]

rows = [('True', 'CropPlanningFixedPlanting'),
        ('True', 'CropPlanningFixedPlantingRotationObserver'),
        ('False', 'CropPlanningFixedPlantingRandomWeather'),
        ('False', 'CropPlanningFixedPlantingRandomWeatherRotationObserver')]

results_mean = results_mean.reindex(rows) / 1000.
results_std = results_std.reindex(rows) / 1000.
results_min = results_min.reindex(rows) / 1000.
results_max = results_max.reindex(rows) / 1000.

tables_dir = PROJECT_PATH.joinpath('tables')
results_mean.to_csv(tables_dir.joinpath('means.csv'), float_format='%.3f')
results_std.to_csv(tables_dir.joinpath('stds.csv'), float_format='%.3f')
round_digit = 2
pm = pd.DataFrame([[' \small{$\pm$ ']*4]*4, index=results_mean.index, columns=results_mean.columns)
curl = pd.DataFrame([['} ']*4]*4, index=results_mean.index, columns=results_mean.columns)
table_string_std = results_mean.round(round_digit).astype('string') + \
                   pm + results_std.round(round_digit).astype('string') + curl
table_string_std.to_csv(tables_dir.joinpath('table_std.csv'), header=False, index=False, sep=str('&'),
                        quoting=csv.QUOTE_NONE)

par = pd.DataFrame([[' \small{(']*4]*4, index=results_mean.index, columns=results_mean.columns)
comma = pd.DataFrame([[',']*4]*4, index=results_mean.index, columns=results_mean.columns)
curl = pd.DataFrame([[')} ']*4]*4, index=results_mean.index, columns=results_mean.columns)
table_string_min_max = results_mean.round(round_digit).astype('string') + \
                       par + results_min.round(round_digit).astype('string') + comma + \
                        results_max.round(round_digit).astype('string') + curl
table_string_min_max.to_csv(tables_dir.joinpath('table_min_max.csv'), header=False, index=False, sep=str('&'),
                            quoting=csv.QUOTE_NONE)

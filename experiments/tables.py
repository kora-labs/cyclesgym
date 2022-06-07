import pandas as pd
import wandb
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

results_mean = runs_df.groupby(['fixed_weather', 'env_class']).mean()[columns]
results_std = runs_df.groupby(['fixed_weather', 'env_class']).std()[columns]
results_mean.to_csv('means.csv', float_format='%.2f')
results_std.to_csv('stds.csv', float_format='%.2f')
Here, we give an overview of how to reproduce the experiments given in the paper. All experiments are tracked using
the wandb library for experiment tracking and reproducibility (http://wandb.ai) that is installed together with the 
cyclesgym package. Before running the expetiments, please create a profile in http://wandb.ai, then logging into wandb 
from the terminal as 

`$ wandb login`

and follow the instructions in the terminal.

### N fertilization environments

It is sufficient to run the script available in the `experiments/fertilization` folder, named `fertilization_experiment.sh`. 
```
$ ./experiments/fertilization/fertilization_experiment.sh
```
The script will lunch the `experiments/fertilization/train.py` script with fixed and random weather generation, and with 
adaptive and non-adaptive policies .... (TODO: and something else?), with 5 fixed seeds.

### Crop planning environments

It is sufficient to run the script available in the `experiments/crop_planning` folder, named `crop_planning_experiment.sh`. 
```
$ ./experiments/crop_planning/crop_planning_experiment.sh
```
The script will lunch the `experiments/crop_planning/train.py` script with fixed and random weather generation, and with 
adaptive and non-adaptive policies, with 5 fixed seeds.

### Custom experiments

For a custom experiment, please refer to the jupyter notebooks available in the `notebook` folder.
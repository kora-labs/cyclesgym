from cyclesgym.utils import run_env
from cyclesgym.envs.crop_plannig import CropPlannig


env = CropPlannig(rotation_crops=['CornSilageRM.90', 'SoybeanMG.5'])
env.reset()

if __name__ == '__main__':
    run_env(env)
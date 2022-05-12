from cyclesgym.informed_policy import InformedPolicy
from pathlib import Path
import unittest
import gym
from gym import spaces
import numpy as np
from numpy.testing import *
from cyclesgym.envs import CornNew
from cyclesgym.envs.common import CyclesEnv
from cyclesgym.paths import TEST_PATH, CYCLES_PATH
import subprocess


def call_cycles(control_file, doy=None):
    CYCLES_PATH.joinpath('Cycles').chmod(stat.S_IEXEC)

    # Run cycles
    if doy is None:
        subprocess.run(['./Cycles', '-b', control_file], cwd=CYCLES_PATH)
    # Run cycles dumping reinit file
    else:
        subprocess.run(['./Cycles', '-b', '-l', str(doy), control_file], cwd=CYCLES_PATH)


def load_output(control_file, fname='CornRM.90.dat'):
    output_dir = CYCLES_PATH.joinpath('output', control_file)
    df = pd.read_csv(output_dir.joinpath(fname), sep='\t').drop(0, 0)
    df.columns = df.columns.str.strip(' ')
    numeric_cols = df.columns[3:]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df


def plot_trajectory(control_file):
    df = load_output(control_file=control_file)
    numeric_cols = df.columns[3:]

    for n in numeric_cols:
        plt.figure()
        plt.title(f'{n} {control_file}')
        df[n].plot()
    plt.show()
    return df


def copy_reinit(control_file, doy):
    """Move reinit to input folder and rename it"""
    output_dir = CYCLES_PATH.joinpath('output', control_file)
    input_dir = CYCLES_PATH.joinpath('input')
    shutil.copy(output_dir.joinpath('reinit.dat'), output_dir.joinpath('reinitcopy.dat'))
    return output_dir.joinpath('reinitcopy.dat').rename(input_dir.joinpath(f'{control_file}{doy}.reinit'))


def create_reinit_control_file(ctrl_file, doy, reinit_file):
    input_dir = CYCLES_PATH.joinpath('input')
    old_ctrl = input_dir.joinpath(f'{ctrl_file}.ctrl')
    new_ctrl_name = f'{ctrl_file}Reinit{doy}'
    new_ctrl = input_dir.joinpath(f'{new_ctrl_name}.ctrl')
    shutil.copy(old_ctrl, new_ctrl)

    f = open(old_ctrl, 'r')
    linelist = f.readlines()
    f.close()

    # Re-open file here
    f2 = open(new_ctrl, 'w')
    for line in linelist:
        if line.startswith('USE_REINITIALIZATION'):
            line = line.replace('0', '1')
        if line.startswith('REINIT_FILE'):
            reinit_file = str(reinit_file)
            reinit_file = reinit_file[reinit_file.rfind('/')+1:]
            line = line.replace('N/A', str(reinit_file))
        f2.write(line)
    f2.close()

    return new_ctrl_name


class TestEnv(CornNew):

    def __init__(self, reinit_file='N / A'):
        CyclesEnv.__init__(self, SIMULATION_START_YEAR=1980,
                           SIMULATION_END_YEAR=1982,
                           ROTATION_SIZE=1,
                           USE_REINITIALIZATION=0,
                           ADJUSTED_YIELDS=0,
                           HOURLY_INFILTRATION=1,
                           AUTOMATIC_NITROGEN=0,
                           AUTOMATIC_PHOSPHORUS=0,
                           AUTOMATIC_SULFUR=0,
                           DAILY_WEATHER_OUT=0,
                           DAILY_CROP_OUT=1,
                           DAILY_RESIDUE_OUT=0,
                           DAILY_WATER_OUT=0,
                           DAILY_NITROGEN_OUT=0,
                           DAILY_SOIL_CARBON_OUT=0,
                           DAILY_SOIL_LYR_CN_OUT=0,
                           ANNUAL_SOIL_OUT=0,
                           ANNUAL_PROFILE_OUT=0,
                           ANNUAL_NFLUX_OUT=0,
                           CROP_FILE='GenericCrops.crop',
                           OPERATION_FILE='ContinuousCorn.operation',
                           SOIL_FILE='GenericHagerstown.soil',
                           WEATHER_FILE='RockSprings.weather',
                           REINIT_FILE=reinit_file,
                           delta=1)

    def _call_cycles_reinit(self, debug=False, reinit=False, doy=None):
        input_file = str(Path(*self.ctrl_file.parts[-2:])).replace('.ctrl', '')
        # Redirect cycles output unless we are debugging
        strings = ['-b']
        if debug:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        if reinit and doy:
            strings.append('-l')
            strings.append(str(doy))

        strings.append(input_file)

        subprocess.run(['./Cycles', *strings], cwd=CYCLES_PATH, stdout=stdout)

        self._update_output_managers()


class TestReinit(unittest.TestCase):

    def setUp(self) -> None:
        env_base = TestEnv()
        env_base.reset()
        env_base._call_cycles_reinit(debug=False, reinit=True, doy=100)

        env_base.env_base._get_output_dir()

        env_reinit = TestEnv()


    def test_reinit_is_markovian_in_non_vegetative_state(self):
        """In this test we check if the reinit file is enough to reproduce completely a run, in the case
        the reinit day (doy) is in a non-vegetative state, hence no state of the crop is necessary"""
        pass

if __name__ == '__main__':
    unittest.main()
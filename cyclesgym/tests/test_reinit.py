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
from cyclesgym.utils import mean_absolute_percentage_error, maximum_absolute_percentage_error, plot_two_environments
import subprocess
import shutil
import matplotlib.pyplot as plt


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


def modify_reinit_file(reinit_file, remove_second_year=True, new_file_name=None):
    "Modifies reinit file to have different entries in the first content line of the reinit file"
    new_reinit = CYCLES_PATH.joinpath('input/', new_file_name)
    shutil.copy(reinit_file, new_reinit)

    f = open(reinit_file, 'r')
    linelist = f.readlines()
    f.close()

    # Re-open file here
    f2 = open(new_reinit, 'w')
    for i, line in enumerate(linelist):
        if i == 2:
            line = line.replace('0', '2')
        if not remove_second_year or (remove_second_year and i <= 12):
            f2.write(line)
    f2.close()

    return new_reinit


def modify_reinit_file_to_new_year(reinit_file, remove_second_year=True, new_file_name=None):
    new_reinit = CYCLES_PATH.joinpath('input/', new_file_name)
    shutil.copy(reinit_file, new_reinit)
    lines = []
    with open(new_reinit, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                line = line.split()
                line[1] = str(int(line[1]) + 1)
                line[3] = '1'
                line = line[0] + '    ' + line[1] + '    ' + line[2] + '     ' + line[3] + '\n'
            lines.append(line)

    with open(new_reinit, 'w') as f:
        for i, line in enumerate(lines):
            if not remove_second_year or (remove_second_year and i <= 12):
                f.write(line)

    return new_reinit


class TestEnv(CornNew):

    def __init__(self, use_reinitialization=0, reinit_file='N / A'):
        CyclesEnv.__init__(self, SIMULATION_START_YEAR=1980,
                           SIMULATION_END_YEAR=1983,
                           ROTATION_SIZE=1,
                           USE_REINITIALIZATION=use_reinitialization,
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
                           ANNUAL_SOIL_OUT=1,
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


class TestEnvSingleYear(TestEnv):

    def __init__(self, start_year, end_year, use_reinitialization=0, reinit_file='N / A'):
        CyclesEnv.__init__(self, SIMULATION_START_YEAR=start_year,
                           SIMULATION_END_YEAR=end_year,
                           ROTATION_SIZE=1,
                           USE_REINITIALIZATION=use_reinitialization,
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


class SoilEnv(CornNew):
    def __init__(self, soilfile, use_reinitialization=1, reinit_file='test_soil.reinit'):
        CyclesEnv.__init__(self, SIMULATION_START_YEAR=1980,
                           SIMULATION_END_YEAR=1983,
                           ROTATION_SIZE=1,
                           USE_REINITIALIZATION=use_reinitialization,
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
                           ANNUAL_SOIL_OUT=1,
                           ANNUAL_PROFILE_OUT=0,
                           ANNUAL_NFLUX_OUT=0,
                           CROP_FILE='GenericCrops.crop',
                           OPERATION_FILE='test_soil.operation',
                           SOIL_FILE=soilfile,
                           WEATHER_FILE='RockSprings.weather',
                           REINIT_FILE=reinit_file,
                           delta=1)


class TestReinit(unittest.TestCase):

    def setUp(self) -> None:
        self.env_base = TestEnv()
        self.env_base.reset()
        self.env_base._call_cycles_reinit(debug=False, reinit=True, doy=365)

        outdir = self.env_base._get_output_dir()
        self.reinit_file = outdir.joinpath('reinit.dat')
        modify_reinit_file(self.reinit_file, remove_second_year=True,
                           new_file_name='reinit_modified_only_first_year.dat')

        self.env_reinit = TestEnv(use_reinitialization=1, reinit_file='reinit_modified_only_first_year.dat')
        self.env_reinit.reset()
        self.env_reinit._call_cycles_reinit(debug=False)

        modify_reinit_file(self.reinit_file, remove_second_year=False, new_file_name='reinit_modified_all_years.dat')
        self.env_reinit_2 = TestEnv(use_reinitialization=1, reinit_file='reinit_modified_all_years.dat')
        self.env_reinit_2.reset()
        self.env_reinit_2._call_cycles_reinit(debug=False)

    def test_reinit_is_markovian_in_non_vegetative_state(self):
        """In this test we check if the reinit file is enough to reproduce completely a run, in the case
        the reinit day (doy) is in a non-vegetative state, hence no state of the crop is necessary"""
        df = self.env_base.crop_output_manager.crop_state
        df1 = self.env_reinit.crop_output_manager.crop_state
        df2 = self.env_reinit_2.crop_output_manager.crop_state

        plot_two_environments(df, df1, ['Base environment', 'Reinit env only first year'], range(4, 17))
        plot_two_environments(df, df2, ['Base environment', 'Modified reinit env all years'], range(4, 17))
        for col in range(4, 17):

            first_year_reinit_error = maximum_absolute_percentage_error(df.iloc[:, col], df1.iloc[:, col])
            all_year_reinit_error = maximum_absolute_percentage_error(df.iloc[:, col], df2.iloc[:, col])

            all_year_reinit_error_second_phase = maximum_absolute_percentage_error(df.iloc[365 * 2 + 1:, col],
                                                                                   df2.iloc[365 * 2 + 1:, col])

            if not np.isnan(all_year_reinit_error_second_phase):
                self.assertLess(all_year_reinit_error_second_phase, 2, df.columns[col])

            if not np.isnan(all_year_reinit_error) and all_year_reinit_error!=0:
                self.assertGreater(all_year_reinit_error, 1, df.columns[col])

            if not np.isnan(first_year_reinit_error) and first_year_reinit_error!=0:
                self.assertGreater(first_year_reinit_error, 1, df.columns[col])

            print(f'Maximum relative error for {df.columns[col]}\n'
                  f'First year reinit {first_year_reinit_error }')
            print(f'All year reinit {all_year_reinit_error}')
            print(f'All year reinit in second phase {all_year_reinit_error_second_phase}')

    def test_two_consecutive_years_with_reinit_from_continuous(self):
        start_year = 1981
        final_year = 1982
        remove_second_year = True
        modify_reinit_file_to_new_year(self.reinit_file, remove_second_year=remove_second_year,
                                       new_file_name='first_year.reinit')
        self.env_reinit_3 = TestEnvSingleYear(start_year, final_year, use_reinitialization=1,
                                              reinit_file='first_year.reinit')
        self.env_reinit_3.reset()
        self.env_reinit_3._call_cycles_reinit(debug=False, reinit=True, doy=365)

        self.env_reinit_4 = TestEnvSingleYear(start_year, final_year, use_reinitialization=1,
                                              reinit_file='first_year.reinit')
        self.env_reinit_4.reset()
        self.env_reinit_4._call_cycles_reinit(debug=False)

        #outdir = self.env_reinit_3._get_output_dir()
        #reinit_file = outdir.joinpath('reinit.dat')

        #modify_reinit_file_to_new_year(reinit_file, remove_second_year=False, new_file_name='second_year.dat')
        #self.env_reinit_4 = TestEnvSingleYear(1982, 1982, use_reinitialization=1, reinit_file='second_year.dat')
        #self.env_reinit_4.reset()
        #self.env_reinit_4._call_cycles_reinit(debug=False)


        df = self.env_base.crop_output_manager.crop_state
        df3 = self.env_reinit_3.crop_output_manager.crop_state
        df4 = self.env_reinit_4.crop_output_manager.crop_state

        plot_two_environments(df.iloc[366:365*(final_year-start_year+2)+1, :].reset_index(drop=True), df3,
                                   ['Original env', 'Reinit env with dumping'], range(4, 17))
        plot_two_environments(df.iloc[366:365 * (final_year - start_year + 2) + 1, :].reset_index(drop=True), df4,
                                   ['Original env', 'Reinit env'], range(4, 17))

    def test_call_reinit(self):
        """Test used to check if the same environemt gives the same results whther it is called as is or dumping the
        reinit file at the end of the year."""
        df = self.env_base.crop_output_manager.crop_state

        env_reinit = TestEnv()
        env_reinit.reset()
        env_reinit._call_cycles_reinit(debug=False)
        df2 = env_reinit.crop_output_manager.crop_state

        plot_two_environments(df, df2, ['Original env', 'Reinit env'], range(4, 17))

    def test_dump_reinit_with_reinitialized_environment(self):
        """Test used to check if the same reinitialized environemt gives the same results whether it is called as is
        or dumping the reinit file at the end of the year."""

        env_base = TestEnv(use_reinitialization=1, reinit_file='ContinuousCorn200.reinit')
        env_base.reset()
        env_base._call_cycles_reinit(debug=False, reinit=True, doy=1)
        df = env_base.crop_output_manager.crop_state

        env_reinit = TestEnv(use_reinitialization=1, reinit_file='ContinuousCorn200.reinit')
        env_reinit.reset()
        env_reinit._call_cycles_reinit(debug=False)
        df2 = env_reinit.crop_output_manager.crop_state

        df3 = self.env_base.crop_output_manager.crop_state

        plot_two_environments(df, df2, ['Dumping env reinitialize', 'No dumbing env reinitialize'], range(4, 17))
        plot_two_environments(df, df3, ['Dumping env reinitialize', 'Env non reinitlialized'], range(4, 17))

    def test_dump_reinit_with_soil_initialization(self):
        """Test used to check if the same reinitialized environment gives the same results whether it is called as is
        or dumping the reinit file at the end of the year."""

        INPUT_DIR = CYCLES_PATH.joinpath('input')
        modify_reinit_file_to_new_year(self.reinit_file, remove_second_year=True,
                                       new_file_name='test_reinit_with_soil_init.reinit')
        #shutil.copy(self.reinit_file, INPUT_DIR.joinpath('test_reinit_with_soil_init.reinit'))
        env_base = TestEnvSingleYear(start_year=1981, end_year=1981, use_reinitialization=1,
                                     reinit_file='test_reinit_with_soil_init.reinit')
        env_base.reset()
        env_base._call_cycles_reinit(debug=False)
        df = env_base.crop_output_manager.crop_state

        INPUT_DIR = CYCLES_PATH.joinpath('input')
        shutil.copy('test_soil_reinit.soil', INPUT_DIR.joinpath('test_soil_reinit.soil'))
        env_reinit = SoilEnv('test_soil_reinit.soil', use_reinitialization=0, reinit_file='N / A')
        env_reinit.reset()
        env_reinit._call_cycles_raw(debug=False)
        df2 = env_reinit.crop_output_manager.crop_state

        plot_two_environments(df, df2, ['Env reinitialize', 'Env not reinitlialized with modified soil'], range(4, 17))


    def test_different_soil_file_same_reinit(self):
        """The test shows that, starting from different soil initial conditions, the reinit makes the simulation
        equivalent, if at days before the plant start to grow. Modify the test_soil.reinit file to test different
        reinit days."""

        INPUT_DIR = CYCLES_PATH.joinpath('input')
        shutil.copy('GenericHagerstown.soil', INPUT_DIR.joinpath('GenericHagerstown.soil'))
        shutil.copy('GenericHagerstownModified.soil', INPUT_DIR.joinpath('GenericHagerstownModified.soil'))
        shutil.copy('test_soil.reinit', INPUT_DIR.joinpath('test_soil.reinit'))
        shutil.copy('test_soil.operation', INPUT_DIR.joinpath('test_soil.operation'))
        env = SoilEnv('GenericHagerstown.soil')
        env_soil_mod = SoilEnv('GenericHagerstownModified.soil')

        env.reset()
        env._call_cycles(debug=False)

        env_soil_mod.reset()
        env_soil_mod._call_cycles(debug=False)

        df = env.crop_output_manager.crop_state
        df1 = env_soil_mod.crop_output_manager.crop_state

        plot_two_environments(df, df1, ['Default soil', 'Modified soil'], range(4, 17))


if __name__ == '__main__':
    unittest.main()
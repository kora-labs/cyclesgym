import pathlib
import shutil
import subprocess
import unittest
import numpy as np

from cyclesgym.envs.corn import CornNew
from cyclesgym.envs.corn_old import CornEnvOld
from cyclesgym.envs.utils import date2ydoy
from cyclesgym.managers import *

from cyclesgym.paths import CYCLES_PATH, TEST_PATH


class TestCornEnvOld(unittest.TestCase):
    def setUp(self):
        self.fnames = ['NCornTest.ctrl',
                       'NCornTest.operation',
                       'NCornTestNoFertilization.ctrl',
                       'NCornTestNoFertilization.operation']
        for n in self.fnames:
            src = TEST_PATH.joinpath(n)
            dest = CYCLES_PATH.joinpath('input', n)
            shutil.copy(src, dest)
        self.custom_sim_id = lambda :'1' # This way the output does not depend on time and can be deleted by teardown

    def tearDown(self):
        for n in self.fnames:
            pathlib.Path(CYCLES_PATH.joinpath('input', n)).unlink()
        try:
            shutil.rmtree(pathlib.Path(CYCLES_PATH.joinpath('output', self.fnames[0].replace('.ctrl', ''))))
            shutil.rmtree(pathlib.Path(CYCLES_PATH.joinpath('output', self.fnames[2].replace('.ctrl', self.custom_sim_id()))))
        except FileNotFoundError:
            pass

    def test_create_sim_operation_file(self):
        env = CornEnvOld(self.fnames[0].replace('.ctrl', ''), delta=7, maxN=150,
                         n_actions=16)
        dest = env._create_sim_operation_file('2')

        # Check the operation file is created
        target_dest = CYCLES_PATH.joinpath('input',
                                           self.fnames[1].replace(
                                              '.operation', '2.operation'))
        assert dest == target_dest

        # Check the operation manager has been updated
        target_keys = [(1, 106, 'PLANTING'),
                       (1, 106, 'TILLAGE'),
                       (1, 106, 'FIXED_FERTILIZATION')]
        assert set(env.op_manager.op_dict.keys()) == set(target_keys)
        assert env.op_manager.op_dict[(1, 106, 'FIXED_FERTILIZATION')][
                   'MASS'] == 0

        # Check the file has been updated
        new_op_manager = OperationManager(dest)
        assert set(new_op_manager.op_dict.keys()) == set(target_keys)
        assert new_op_manager.op_dict[(1, 106, 'FIXED_FERTILIZATION')][
                   'MASS'] == 0
        # Delete file
        dest.unlink()

    def test_create_sim_ctrl_file(self):
        env = CornEnvOld(self.fnames[0].replace('.ctrl', ''), delta=7, maxN=150,
                         n_actions=16)
        dest = env._create_sim_ctrl_file('new_operation2.operation', '2')
        target_dest = CYCLES_PATH.joinpath('input',
                                           self.fnames[0].replace(
                                              '.ctrl', '2.ctrl'))
        assert dest == target_dest
        assert env.ctrl == Path(self.fnames[0].replace('.ctrl', '2.ctrl'))
        assert env.ctrl_manager.ctrl_dict['OPERATION_FILE'] == \
        'new_operation2.operation'

        # Delete file
        dest.unlink()


class TestCornEnv(unittest.TestCase):
    def setUp(self):
        # Copy test files in cycles input directory
        self.fnames = ['NCornTest.ctrl',
                       'NCornTest.operation',
                       'NCornTestNoFertilization.operation']
        for n in self.fnames:
            shutil.copy(TEST_PATH.joinpath(n),
                        CYCLES_PATH.joinpath('input', n))

        # This way the output does not depend on time and can be deleted by teardown
        self.custom_sim_id = lambda : '1'

    def tearDown(self):

        # Remove all files copied to input folder
        for n in self.fnames:
            pathlib.Path(CYCLES_PATH.joinpath('input', n)).unlink()
        # Remove the output of simulation started manually
        try:
            shutil.rmtree(pathlib.Path(CYCLES_PATH.joinpath('output', self.fnames[0].replace('.ctrl', ''))))
        except FileNotFoundError:
            pass

    def test_manual_sim_comparison(self):
        """
        Check the manual and env simulation are the same for same management
        and different for different management.
        """
        # Start normal simulation and parse results
        base_ctrl_man = self.fnames[0].replace('.ctrl', '')
        self._call_cycles(base_ctrl_man)
        crop_man = CropManager(
            CYCLES_PATH.joinpath('output', base_ctrl_man, 'CornRM.90.dat'))

        # Start gym env
        operation_file = self.fnames[2]
        env = CornNew(delta=1, maxN=150, n_actions=16,
                      operation_file=operation_file)

        # Run simulation with same management and compare
        env.reset()
        while True:
            _, doy = date2ydoy(env.date)
            a = 15 if doy == 106 else 0
            _, _, done, _ = env.step(a)
            if done:
                break
        crop_output_file = env._get_output_dir().joinpath('CornRM.90.dat')
        crop_env = CropManager(crop_output_file)
        assert crop_env.crop_state.equals(crop_man.crop_state)

        # Run simulation with different management and compare
        env.reset()
        while True:
            _, doy = date2ydoy(env.date)
            a = 15 if doy == 107 else 0
            _, _, done, _ = env.step(a)
            if done:
                break
        crop_output_file = env._get_output_dir().joinpath('CornRM.90.dat')
        crop_env = CropManager(crop_output_file)
        assert not crop_env.crop_state.equals(crop_man.crop_state)

    @staticmethod
    def _call_cycles(ctrl):
        subprocess.run(['./Cycles', '-b', ctrl], cwd=CYCLES_PATH)


if __name__ == '__main__':
    unittest.main()

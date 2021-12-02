import pathlib
from pathlib import Path
import unittest
import subprocess
import shutil
from cyclesgym.env import CornEnv
from cyclesgym.managers import *

from cyclesgym.cycles_config import CYCLES_DIR
# CYCLES_DIR = Path.cwd().parent.parent.joinpath('cycles')


class TestCornEnv(unittest.TestCase):
    def setUp(self):
        self.fnames = ['NCornTest.ctrl', 'NCornTest.operation',
                       'NCornTestEnv.ctrl', 'NCornTestEnv.operation']
        for n in self.fnames:
            src = pathlib.Path.cwd().joinpath(n)
            dest = CYCLES_DIR.joinpath('input', n)
            shutil.copy(src, dest)
        self.env = CornEnv(self.fnames[2].replace('.ctrl', ''), delta=7, maxN=150, n_actions=16)
        self.custom_sim_id = lambda :'1'
        self.env._create_sim_id = \
            self.custom_sim_id  # This way the output does not depend on time and can be deleted by teardown

    def tearDown(self):
        for n in self.fnames:
            pathlib.Path(CYCLES_DIR.joinpath('input', n)).unlink()
        try:
            shutil.rmtree(pathlib.Path(CYCLES_DIR.joinpath('output', self.fnames[0].replace('.ctrl', ''))))
            shutil.rmtree(pathlib.Path(CYCLES_DIR.joinpath('output', self.fnames[2].replace('.ctrl', self.custom_sim_id()))))
        except FileNotFoundError:
            pass

    def test_equal(self):
        # Start normal simulation and parse results
        self._call_cycles(self.fnames[0].replace('.ctrl', ''))
        crop_from_sim = CropManager(CYCLES_DIR.joinpath('output',
                                                        self.fnames[0].replace('.ctrl', ''),
                                                        'CornRM.90.dat'))

        self.env.reset()
        for i in range(365):
            a = 15 if i == 15 else 0
            _, _, done, _ = self.env.step(a)
            if done:
                break
        crop_from_env = CropManager(CYCLES_DIR.joinpath('output',
                                                        self.fnames[2].replace('.ctrl', self.custom_sim_id()),
                                                        'CornRM.90.dat'))
        assert crop_from_env.crop_state.equals(crop_from_sim.crop_state)

    def test_different(self):
        # Start normal simulation and parse results
        self._call_cycles(self.fnames[0].replace('.ctrl', ''))
        crop_from_sim = CropManager(CYCLES_DIR.joinpath('output',
                                                        self.fnames[0].replace('.ctrl', ''),
                                                        'CornRM.90.dat'))
    #
        self.env.reset()
        for i in range(365):
            a = 0
            _, _, done, _ = self.env.step(a)
            if done:
                break
        crop_from_env = CropManager(CYCLES_DIR.joinpath('output',
                                                        self.fnames[2].replace('.ctrl', self.custom_sim_id()),
                                                        'CornRM.90.dat'))
        assert not crop_from_env.crop_state.equals(crop_from_sim.crop_state)

    @staticmethod
    def _call_cycles(ctrl):
        subprocess.run(['./Cycles', '-b', ctrl], cwd=CYCLES_DIR)

    def test_check_new_action(self):
        pass

    def test_update_operation(self):
        base_op = {'SOURCE': 'UreaAmmoniumNitrate', 'MASS': 80, 'FORM': 'Liquid', 'METHOD': 'Broadcast', 'LAYER': 1,
                   'C_Organic': 0.5, 'C_Charcoal': 0, 'N_Organic': 0, 'N_Charcoal': 0, 'N_NH4': 0, 'N_NO3': 0,
                   'P_Organic': 0, 'P_CHARCOAL': 0, 'P_INORGANIC': 0, 'K': 0.5, 'S': 0}
        target_new_op = base_op.copy()
        target_new_op.update({'MASS': 100, 'C_Organic': 0.4, 'K': 0.4, 'N_NH4': 0.15, 'N_NO3': 0.05})
        new_op = self.env._udpate_operation(base_op, N_mass=20, mode='increment')
        assert target_new_op == new_op

        base_op = {'SOURCE': 'UreaAmmoniumNitrate', 'MASS': 80, 'FORM': 'Liquid', 'METHOD': 'Broadcast', 'LAYER': 1,
                   'C_Organic': 0.25, 'C_Charcoal': 0, 'N_Organic': 0, 'N_Charcoal': 0, 'N_NH4': 0.25, 'N_NO3': 0.25,
                   'P_Organic': 0, 'P_CHARCOAL': 0, 'P_INORGANIC': 0, 'K': 0.25, 'S': 0}
        target_new_op = base_op.copy()
        target_new_op.update({'MASS': 40, 'C_Organic': 0.5, 'K': 0.5, 'N_NH4': 0, 'N_NO3': 0})
        new_op = self.env._udpate_operation(base_op, N_mass=0, mode='absolute')
        assert target_new_op == new_op

if __name__ == '__main__':
    unittest.main()
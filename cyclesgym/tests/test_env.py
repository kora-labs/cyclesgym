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
        shutil.rmtree(pathlib.Path(CYCLES_DIR.joinpath('output', self.fnames[0].replace('.ctrl', ''))))
        shutil.rmtree(pathlib.Path(CYCLES_DIR.joinpath('output', self.fnames[2].replace('.ctrl', self.custom_sim_id()))))

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
            a = 15
            _, _, done, _ = self.env.step(a)
            if done:
                break
        crop_from_env = CropManager(CYCLES_DIR.joinpath('output',
                                                        self.fnames[2].replace('.ctrl', self.custom_sim_id()),
                                                        'CornRM.90.dat'))
        # Plot
        # columns = [['THERMAL TIME', 'CUM. BIOMASS'],
        #            ['TOTAL N', 'N STRESS']]
        # crop_from_sim.plot(columns=columns)
        # crop_from_env.plot(columns=columns)
        assert not crop_from_env.crop_state.equals(crop_from_sim.crop_state)

    @staticmethod
    def _call_cycles(ctrl):
        subprocess.run(['./Cycles', '-b', ctrl], cwd=CYCLES_DIR)


if __name__ == '__main__':
    unittest.main()
import pathlib
from pathlib import Path
import unittest
import subprocess
import shutil
from cyclesgym.env import NWinterWheatEnv
from cyclesgym.managers import *

from cyclesgym.cycles_config import CYCLES_DIR
# CYCLES_DIR = Path.cwd().parent.parent.joinpath('cycles')


class TestNWinterWheat(unittest.TestCase):
    def setUp(self):
        self.fnames = ['NWinterWheatTest.ctrl', 'NWinterWheatTest.operation',
                       'NWinterWheatTestEnv.ctrl', 'NWinterWheatTestEnv.operation']
        for n in self.fnames:
            src = pathlib.Path.cwd().joinpath(n)
            dest = CYCLES_DIR.joinpath('input', n)
            shutil.copy(src, dest)

    def tearDown(self):
        for n in self.fnames:
            pathlib.Path(CYCLES_DIR.joinpath('input', n)).unlink()
        shutil.rmtree(pathlib.Path(CYCLES_DIR.joinpath('output', self.fnames[0].replace('.ctrl', ''))))
        shutil.rmtree(pathlib.Path(CYCLES_DIR.joinpath('output', self.fnames[2].replace('.ctrl', ''))))

    def test_equal(self):
        # Start normal simulation and parse results
        self._call_cycles('NWinterWheatTest')
        crop_from_sim = CropManager(CYCLES_DIR.joinpath('output', 'NWinterWheatTest', 'CornRM.90.dat'))

        env = NWinterWheatEnv(CYCLES_DIR.joinpath('input', 'NWinterWheatTestEnv'), delta=7, maxN=150, n_actions=16)
        env.reset()
        for i in range(int(365/7)):
            a = 15 if i == 15 else 0
            env.step(a)
        crop_from_env = CropManager(CYCLES_DIR.joinpath('output', 'NWinterWheatTestEnv', 'CornRM.90.dat'))
        assert crop_from_env.crop_state.equals(crop_from_sim.crop_state)

    def test_different(self):
        # Start normal simulation and parse results
        self._call_cycles('NWinterWheatTest')
        crop_from_sim = CropManager(CYCLES_DIR.joinpath('output', 'NWinterWheatTest', 'CornRM.90.dat'))

        env = NWinterWheatEnv(CYCLES_DIR.joinpath('input', 'NWinterWheatTestEnv'), delta=7, maxN=150, n_actions=16)
        env.reset()
        for i in range(int(365 / 7)):
            env.step(15)
        crop_from_env = CropManager(CYCLES_DIR.joinpath('output', 'NWinterWheatTestEnv', 'CornRM.90.dat'))

        columns = [['THERMAL TIME', 'CUM. BIOMASS'],
                   ['TOTAL N', 'N STRESS']]
        crop_from_sim.plot(columns=columns)
        crop_from_env.plot(columns=columns)
        assert not crop_from_env.crop_state.equals(crop_from_sim.crop_state)



    @staticmethod
    def _call_cycles(ctrl):
        subprocess.run(['./Cycles', '-b', ctrl], cwd=CYCLES_DIR)


if __name__ == '__main__':
    unittest.main()
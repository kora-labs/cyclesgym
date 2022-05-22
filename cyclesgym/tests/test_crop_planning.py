from cyclesgym.utils import run_env
from cyclesgym.envs.crop_plannig import CropPlannig
import unittest
import shutil
import subprocess

from cyclesgym.managers import *
from cyclesgym.utils import diff_pd
from cyclesgym.paths import CYCLES_PATH, TEST_PATH


class TestCropPlanning(unittest.TestCase):

    @staticmethod
    def _call_cycles(ctrl):
        subprocess.run(['./Cycles', '-b', ctrl], cwd=CYCLES_PATH)

    def setUp(self):
        self.fnames = ['CropPlanningTest.ctrl',
                       'CropPlanningTest.operation']
        for n in self.fnames:
            src = TEST_PATH.joinpath(n)
            dest = CYCLES_PATH.joinpath('input', n)
            shutil.copy(src, dest)
        self.custom_sim_id = lambda :'1' # This way the output does not depend on time and can be deleted by teardown

    def test_equal(self):
        # Start normal simulation and parse results
        self._call_cycles(self.fnames[0].replace('.ctrl', ''))
        crop_from_sim_1 = CropManager(CYCLES_PATH.joinpath('output',
                                                         self.fnames[0].replace('.ctrl', ''),
                                                        'CornSilageRM.90.dat'))
        crop_from_sim_2 = CropManager(CYCLES_PATH.joinpath('output',
                                                           self.fnames[0].replace('.ctrl', ''),
                                                           'SoybeanMG.5.dat'))

        env = CropPlannig(start_year=1980, end_year=1990, rotation_crops=['CornSilageRM.90', 'SoybeanMG.5'])
        env._create_sim_id = self.custom_sim_id

        env.reset()
        year = 0

        policy = [(0, 115/365, 115/365, 1.0), (1, 120/365, 120/365, 1.0) ]

        while True:
            a = policy[year % 2]
            _, _, done, _ = env.step(a)
            year += 1
            if done:
                break

        crop_from_env_1 = env.crop_output_manager[0]
        crop_from_env_2 = env.crop_output_manager[1]

        assert crop_from_env_1.crop_state.equals(crop_from_sim_1.crop_state), \
            diff_pd(crop_from_env_1.crop_state, crop_from_sim_1.crop_state)
        assert crop_from_env_2.crop_state.equals(crop_from_sim_2.crop_state), \
            diff_pd(crop_from_env_2.crop_state, crop_from_sim_2.crop_state)


if __name__ == '__main__':
    unittest.main()

    env = CropPlannig(start_year=1980, end_year=2000, rotation_crops=['CornSilageRM.90', 'SoybeanMG.5'])
    env.reset()
    run_env(env)
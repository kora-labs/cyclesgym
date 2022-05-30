from cyclesgym.utils import run_env
from cyclesgym.envs.crop_planning import CropPlanningFixedPlanting
import unittest
import shutil
import subprocess

from cyclesgym.managers import *
from cyclesgym.utils import diff_pd
from cyclesgym.paths import CYCLES_PATH, TEST_PATH
import time


class CropPlanningBaselines(object):

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
        self.custom_sim_id = lambda: '1'  # This way the output does not depend on time and can be deleted by teardown

    def _test_policy(self, policy):
        env = CropPlanningFixedPlanting(start_year=2000, end_year=2016, rotation_crops=['CornRM.100',
                                                                                        'SoybeanMG.3'])
        env.reset()
        year = 0

        start = time.time()
        rewards = []
        while True:
            a = policy[year]
            _, r, done, _ = env.step(a)
            rewards.append(r)
            year += 1
            if done:
                break

        crop_from_env_1 = env.crop_output_manager[0]
        crop_from_env_2 = env.crop_output_manager[1]
        print(time.time() - start)
        return crop_from_env_1, crop_from_env_2, rewards

    def test_equal(self):
        rotation_policy = [(1, 2), (0, 1)]*11
        only_soy = [(1, 2)]*21
        only_corn = [(0, 1)]*21
        long_rotation = [(1, 2), (1, 2), (1, 2), (0, 1)]*7

        crop_from_env_1, crop_from_env_2, rewards = self._test_policy(long_rotation)
        print(sum(rewards))

        crop_from_env_1, crop_from_env_2, rewards = self._test_policy(rotation_policy)
        print(sum(rewards))

        crop_from_env_1, crop_from_env_2, rewards = self._test_policy(only_soy)
        print(sum(rewards))

        crop_from_env_1, crop_from_env_2, rewards = self._test_policy(only_corn)
        print(sum(rewards))


if __name__ == '__main__':
    CropPlanningBaselines().test_equal()
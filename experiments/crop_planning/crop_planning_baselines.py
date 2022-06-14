from cyclesgym.utils import run_env
from cyclesgym.envs.crop_planning import CropPlanningFixedPlanting
import unittest
import shutil
import subprocess
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

    def _test_policy(self, policy, start, end, weather_file):
        env = CropPlanningFixedPlanting(start_year=start, end_year=end, weather_file=weather_file,
                                        rotation_crops=['CornRM.100', 'SoybeanMG.3'])
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

    def test_equal(self, start, end, weather_file):
        long_rotation = [(1, 2), (1, 2), (0, 1)] * (end - start)
        short_rotation = [(1, 2), (0, 1)] * (end - start + 1)
        only_soy = [(1, 2)] * (end - start + 1)
        only_corn = [(0, 1)] * (end - start + 1)

        def test_policy(policy, text):
            crop_from_env_1, crop_from_env_2, rewards = self._test_policy(policy,  start, end, weather_file)
            print(text, start, end, weather_file, sum(rewards) / (end - start + 1))

        test_policy(long_rotation, 'long_rotation')
        test_policy(short_rotation, 'short rotation')
        test_policy(only_soy, 'only soy')
        test_policy(only_corn, 'only_corn')


if __name__ == '__main__':
    train_start_year = 1980
    train_end_year = 1998
    test_end_year = 2016

    weather_train_file = 'RockSprings.weather'
    weather_test_file = 'NewHolland.weather'

    CropPlanningBaselines().test_equal(1980, 1998, weather_train_file)
    CropPlanningBaselines().test_equal(1998, 2016, weather_train_file)
    CropPlanningBaselines().test_equal(1980, 1998, weather_test_file)
    CropPlanningBaselines().test_equal(1980, 2015, weather_test_file)
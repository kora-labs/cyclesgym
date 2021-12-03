import pathlib
from pathlib import Path
import unittest
import subprocess
import shutil
import numpy as np
from cyclesgym.env import CornEnv
from cyclesgym.managers import *

from cyclesgym.cycles_config import CYCLES_DIR


class TestCornEnv(unittest.TestCase):
    def setUp(self):
        self.fnames = ['NCornTest.ctrl',
                       'NCornTest.operation',
                       'NCornTestNoFertilization.ctrl',
                       'NCornTestNoFertilization.operation']
        for n in self.fnames:
            src = pathlib.Path.cwd().joinpath(n)
            dest = CYCLES_DIR.joinpath('input', n)
            shutil.copy(src, dest)
        self.custom_sim_id = lambda :'1' # This way the output does not depend on time and can be deleted by teardown

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
        env = CornEnv(self.fnames[2].replace('.ctrl', ''), delta=7, maxN=150,
                      n_actions=16)
        env._create_sim_id = self.custom_sim_id

        env.reset()
        week = 0
        while True:
            a = 15 if week == 15 else 0
            _, _, done, _ = env.step(a)
            week += 1
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
        env = CornEnv(self.fnames[2].replace('.ctrl', ''), delta=7, maxN=150,
                      n_actions=16)
        env._create_sim_id = self.custom_sim_id

        env.reset()
        while True:
            a = 0
            _, _, done, _ = env.step(a)
            if done:
                break
        crop_from_env = CropManager(CYCLES_DIR.joinpath('output',
                                                        self.fnames[2].replace('.ctrl', self.custom_sim_id()),
                                                        'CornRM.90.dat'))
        assert not crop_from_env.crop_state.equals(crop_from_sim.crop_state)

    def test_create_sim_operation_file(self):
        env = CornEnv(self.fnames[0].replace('.ctrl', ''), delta=7, maxN=150,
                      n_actions=16)
        dest = env._create_sim_operation_file('2')

        # Check the operation file is created
        target_dest = CYCLES_DIR.joinpath('input',
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
        env = CornEnv(self.fnames[0].replace('.ctrl', ''), delta=7, maxN=150,
                      n_actions=16)
        dest = env._create_sim_ctrl_file('new_operation2.operation', '2')
        target_dest = CYCLES_DIR.joinpath('input',
                                          self.fnames[0].replace(
                                              '.ctrl', '2.ctrl'))
        assert dest == target_dest
        assert env.ctrl == Path(self.fnames[0].replace('.ctrl', '2.ctrl'))
        assert env.ctrl_manager.ctrl_dict['OPERATION_FILE'] == \
        'new_operation2.operation'

        # Delete file
        dest.unlink()

    def test_update_operation(self):
        env = CornEnv(self.fnames[2].replace('.ctrl', ''), delta=7, maxN=150,
                      n_actions=16)
        env._create_sim_id = self.custom_sim_id

        base_op = {'SOURCE': 'UreaAmmoniumNitrate', 'MASS': 80, 'FORM': 'Liquid', 'METHOD': 'Broadcast', 'LAYER': 1,
                   'C_Organic': 0.5, 'C_Charcoal': 0, 'N_Organic': 0, 'N_Charcoal': 0, 'N_NH4': 0, 'N_NO3': 0,
                   'P_Organic': 0, 'P_CHARCOAL': 0, 'P_INORGANIC': 0, 'K': 0.5, 'S': 0}
        target_new_op = base_op.copy()
        target_new_op.update({'MASS': 100, 'C_Organic': 0.4, 'K': 0.4, 'N_NH4': 0.15, 'N_NO3': 0.05})
        new_op = env._udpate_operation(base_op, N_mass=20, mode='increment')
        assert target_new_op == new_op

        base_op = {'SOURCE': 'UreaAmmoniumNitrate', 'MASS': 80, 'FORM': 'Liquid', 'METHOD': 'Broadcast', 'LAYER': 1,
                   'C_Organic': 0.25, 'C_Charcoal': 0, 'N_Organic': 0, 'N_Charcoal': 0, 'N_NH4': 0.25, 'N_NO3': 0.25,
                   'P_Organic': 0, 'P_CHARCOAL': 0, 'P_INORGANIC': 0, 'K': 0.25, 'S': 0}
        target_new_op = base_op.copy()
        target_new_op.update({'MASS': 40, 'C_Organic': 0.5, 'K': 0.5, 'N_NH4': 0, 'N_NO3': 0})
        new_op = env._udpate_operation(base_op, N_mass=0, mode='absolute')
        assert target_new_op == new_op

    def test_reset(self):
        pass

    def test_check_new_action(self):
        pass

    def test_step(self):
        pass

    def test_compute_obs(self):
        env = CornEnv(self.fnames[2].replace('.ctrl', ''), delta=7, maxN=150,
                      n_actions=16)
        env.weather_manager = WeatherManager(Path.cwd().joinpath(
            'DummyWeather.weather'))
        env.crop_manager = CropManager(Path.cwd().joinpath('DummyCrop.dat'))
        obs = env.compute_obs(1980, 1)
        target_obs = np.concatenate((np.arange(14), np.array([40.6875, 0, 10]),
                                     np.arange(7)))
        assert np.all(obs == target_obs)

    def test_compute_reward(self):
        env = CornEnv(self.fnames[2].replace('.ctrl', ''), delta=7, maxN=150,
                      n_actions=16)
        env.season_manager = SeasonManager(Path.cwd().joinpath('DummySeason.dat'))
        r = env.compute_reward()
        assert r == 1

    def test_implement_action(self):
        pass

    def test_move_sim_specific_files(self):
        pass

    @staticmethod
    def _call_cycles(ctrl):
        subprocess.run(['./Cycles', '-b', ctrl], cwd=CYCLES_DIR)


if __name__ == '__main__':
    unittest.main()
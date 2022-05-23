import unittest
import numpy as np
import shutil
from pathlib import Path

from cyclesgym.envs.implementers import Fertilizer
from cyclesgym.managers import OperationManager
from cyclesgym.envs.utils import ydoy2date

from cyclesgym.paths import TEST_PATH


class TestFertilizer(unittest.TestCase):
    def setUp(self) -> None:
        # Copy file to make sure we do not modify the original one
        src = TEST_PATH.joinpath('NCornTest.operation')
        self.op_fname = TEST_PATH.joinpath('NCornTest_copy.operation')
        shutil.copy(src, self.op_fname)

        # Init observer
        self.op_manager = OperationManager(self.op_fname)

        # Init Fertilizer
        self.fert = Fertilizer(operation_manager=self.op_manager,
                               operation_fname=self.op_fname,
                               affected_nutrients= ['N_NH4', 'N_NO3'],
                               start_year=1980)

    def tearDown(self) -> None:
        # Remove copied operation file
        self.op_fname.unlink()

    def test_new_action(self):
        # No action on a day where nothing used to happen => Not new
        assert not self.fert._is_new_action(
            year=1, doy=105, new_masses={'N_NH4': 0, 'N_NO3': 0})

        # Same fertilization action as the one already in the file => Not new
        assert not self.fert._is_new_action(
            year=1, doy=106, new_masses={'N_NH4': 112.5, 'N_NO3': 37.5})

        # No action on a day when we used to fertilize => New
        assert self.fert._is_new_action(
            year=1, doy=106, new_masses={'N_NH4': 0, 'N_NO3': 0})

        # Fertilize on a day when we used to do nothing => New
        assert self.fert._is_new_action(
            year=1, doy=105, new_masses={'N_NH4': 112.5, 'N_NO3': 37.5})

    def test_update_operation(self):
        # Incremental mode
        base_op = {'SOURCE': 'UreaAmmoniumNitrate', 'MASS': 80,
                   'FORM': 'Liquid', 'METHOD': 'Broadcast', 'LAYER': 1,
                   'C_Organic': 0.5, 'C_Charcoal': 0., 'N_Organic': 0.,
                   'N_Charcoal': 0., 'N_NH4': 0., 'N_NO3': 0.,
                   'P_Organic': 0., 'P_CHARCOAL': 0., 'P_INORGANIC': 0.,
                   'K': 0.5, 'S': 0.}

        year = 1
        doy = 106
        new_op = self.fert._update_operation(
            year=1, doy=106, old_op=base_op,
            new_masses={'N_NH4': 15, 'N_NO3': 5}, mode='increment')

        target_new_op = base_op.copy()
        target_new_op.update(
            {'MASS': 100., 'C_Organic': 0.4, 'K': 0.4, 'N_NH4': 0.15,
             'N_NO3': 0.05})

        assert target_new_op == new_op[(year, doy, 'FIXED_FERTILIZATION')]

        # Absolute mode
        base_op = {'SOURCE': 'UreaAmmoniumNitrate', 'MASS': 80,
                   'FORM': 'Liquid', 'METHOD': 'Broadcast', 'LAYER': 1,
                   'C_Organic': 0.25, 'C_Charcoal': 0, 'N_Organic': 0,
                   'N_Charcoal': 0, 'N_NH4': 0.25, 'N_NO3': 0.25,
                   'P_Organic': 0, 'P_CHARCOAL': 0, 'P_INORGANIC': 0,
                   'K': 0.25, 'S': 0}

        new_op = self.fert._update_operation(
            year=1, doy=106, old_op=base_op,
            new_masses={'N_NH4': 0, 'N_NO3': 0}, mode='absolute')

        target_new_op = base_op.copy()
        target_new_op.update(
            {'MASS': 40, 'C_Organic': 0.5, 'K': 0.5, 'N_NH4': 0, 'N_NO3': 0})

        assert target_new_op == new_op[(year, doy, 'FIXED_FERTILIZATION')]

    def test_implement_with_collision(self):
        operations = self.fert.operation_manager.op_dict.copy()
        target_op = {'SOURCE': 'Unknown', 'MASS': 20.0,
                     'FORM': 'Unknown', 'METHOD': 'Unknown', 'LAYER': 1.,
                     'C_Organic': 0., 'C_Charcoal': 0., 'N_Organic': 0.,
                     'N_Charcoal': 0., 'N_NH4': 0.75, 'N_NO3': 0.25,
                     'P_Organic': 0., 'P_CHARCOAL': 0., 'P_INORGANIC': 0.,
                     'K': 0., 'S': 0.}
        operations.update({(1, 106, 'FIXED_FERTILIZATION'): target_op})

        self.fert.implement_action(date=ydoy2date(1980, 106),
                                   operation_details={'N_NH4': 15.,
                                                      'N_NO3': 5.})

        # Check manager is equal
        assert self.fert.operation_manager.op_dict == operations

        # Check file is equal
        new_manager = OperationManager(self.fert.operation_fname)
        assert new_manager.op_dict == self.fert.operation_manager.op_dict

    def test_implement_no_collision(self):
        operations = self.fert.operation_manager.op_dict.copy()
        target_op = {'SOURCE': 'Unknown', 'MASS': 20.0,
                     'FORM': 'Unknown', 'METHOD': 'Unknown', 'LAYER': 1.,
                     'C_Organic': 0., 'C_Charcoal': 0., 'N_Organic': 0.,
                     'N_Charcoal': 0., 'N_NH4': 0.75, 'N_NO3': 0.25,
                     'P_Organic': 0., 'P_CHARCOAL': 0., 'P_INORGANIC': 0.,
                     'K': 0., 'S': 0.}
        operations.update({(1, 105, 'FIXED_FERTILIZATION'): target_op})

        self.fert.implement_action(date=ydoy2date(1980, 105),
                                   operation_details={'N_NH4': 15.,
                                                      'N_NO3': 5.})

        # Check manager is equal
        assert self.fert.operation_manager.op_dict == operations

        # Check file is equal
        new_manager = OperationManager(self.fert.operation_fname)
        assert new_manager.op_dict == self.fert.operation_manager.op_dict



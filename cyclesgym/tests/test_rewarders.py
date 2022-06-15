import datetime
import unittest

from cyclesgym.envs.rewarders import CropRewarder
from cyclesgym.utils.pricing_utils import crop_prices
from cyclesgym.managers import *

from cyclesgym.utils.paths import TEST_PATH


class TestRewarders(unittest.TestCase):
    def setUp(self) -> None:
        # Init managers
        self.season_manager = SeasonManager(
            TEST_PATH.joinpath('DummySeason.dat'))

        # Init observers
        self.corn_rewarder = CropRewarder(
            season_manager=self.season_manager,
            crop_name='CornRM.90'
        )

    def test_crop_r(self):
        # Actual harvest date: 1980-09-08

        # Test during harvest
        date = datetime.date(year=1980, month=9, day=12)
        r = self.corn_rewarder.compute_reward(date=date, delta=5)
        target_r = 3 * crop_prices['CornRM.90'][1980]
        assert r == target_r

        # Test out of harvest
        r = self.corn_rewarder.compute_reward(date=date, delta=3)
        assert r == 0


if __name__ == '__main__':
    unittest.main()
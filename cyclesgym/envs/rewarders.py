from datetime import timedelta
from cyclesgym.envs.utils import date2ydoy, ydoy2date
from cyclesgym.pricing_utils import crop_prices, N_price_dollars_per_kg, crop_type
import datetime

__all__ = ['CropRewarder']


class CropRewarder(object):

    def __init__(self, season_manager, crop_name):
        self.season_manager = season_manager
        self.crop_name = crop_name
        self.dollars_per_tonne = crop_prices[self.crop_name]
        self.yield_column = crop_type[self.crop_name]

    def _harvest_profit(self, date, delta, action=None):
        # Date of previous time step
        del action

        previous_date = date - timedelta(days=delta)
        y_prev, doy_prev = date2ydoy(previous_date)

        # Did we harvest between this and previous time step?
        df = self.season_manager.season_df
        harverst_df = df.loc[(df['YEAR'] == y_prev) & (df['CROP'] == self.crop_name)]
        #harvest on 244. plant on 110
        harvest_dollars_per_hectare = 0
        if not harverst_df.empty:
            harverst_doy = harverst_df.iloc[0]['DOY']
            harvest_date = ydoy2date(y_prev, harverst_doy)

            if previous_date < harvest_date <= date:
                # Compute harvest profit
                dollars_per_tonne = self.dollars_per_tonne[y_prev]
                harvest = float(harverst_df[self.yield_column])  # Metric tonne per hectar
                harvest_dollars_per_hectare = harvest * dollars_per_tonne
        return harvest_dollars_per_hectare
    
    def compute_reward(self, date, delta, action=None):
        return self._harvest_profit(date, delta, action=action)


class NProfitabilityRewarder(object):

    def compute_reward(self, date, delta, action=None):
        Nkg_per_heactare = action
        assert Nkg_per_heactare >= 0, f'We cannot have negative fertilization'
        y, doy = date2ydoy(date)
        N_dollars_per_hectare = Nkg_per_heactare * N_price_dollars_per_kg[y]
        return -N_dollars_per_hectare


def compound_rewarder(rewarder_list: list):

    class Compound(object):
        def __init__(self, rewarder_list):
            self.rewarder_list = rewarder_list

        def compute_reward(self, date: datetime.date, delta, action=None):
            return sum([r.compute_reward(date, delta, action=action) for r in self.rewarder_list])

    return Compound(rewarder_list)



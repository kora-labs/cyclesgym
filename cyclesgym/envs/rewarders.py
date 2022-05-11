from datetime import timedelta
from cyclesgym.envs.utils import date2ydoy, ydoy2date

__all__ = ['CornNProfitabilityRewarder']

# Conversion rate for corn from bushel to metric ton from https://grains.org/markets-tools-data/tools/converting-grain-units/
BUSHEL_PER_TONNE = 39.3680


class CornNProfitabilityRewarder(object):
    # Avg anhydrous ammonia cost in 2020 from https://farmdocdaily.illinois.edu/2021/08/2021-fertilizer-price-increases-in-perspective-with-implications-for-2022-costs.html
    # Computed as 496 * 0.001 ($/ton * ton/kg)
    N_dollars_per_kg = {y: 496 * 0.001 for y in range(1980, 2020)}

    # Avg US price of corn for 2020 from https://quickstats.nass.usda.gov/results/BA8CCB81-A2BB-3C5C-BD23-DBAC365C7832
    dollars_per_bushel = {y: 4.53 for y in range(1980, 2020)}

    def __init__(self, season_manager):
        self.season_manager = season_manager

    def _N_penalty(self, Nkg_per_heactare, date):
        assert Nkg_per_heactare >= 0, f'We cannot have negative fertilization'
        y, doy = date2ydoy(date)
        N_dollars_per_hectare = Nkg_per_heactare * self.N_dollars_per_kg[y]
        return -N_dollars_per_hectare

    def _harvest_profit(self, date, delta):
        # Date of previous time step

        previous_date = date - timedelta(days=delta)
        y_prev, doy_prev = date2ydoy(previous_date)

        # Did we harvest between this and previous time step?
        df = self.season_manager.season_df
        harverst_df = df.loc[df['YEAR'] == y_prev]
        harverst_doy = harverst_df.iloc[0]['DOY']
        harvest_date = ydoy2date(y_prev, harverst_doy)

        if previous_date <= harvest_date <= date:
            # Compute harvest profit
            dollars_per_tonne = self.dollars_per_bushel[y_prev] * \
                                BUSHEL_PER_TONNE
            harvest = harverst_df['GRAIN YIELD']  # Metric tonne per hectar
            harvest_dollars_per_hectare = harvest * dollars_per_tonne
            return harvest_dollars_per_hectare
        else:
            return 0

    def compute_reward(self, Nkg_per_heactare, date, delta):
        r = 0
        r += self._N_penalty(Nkg_per_heactare, date)
        r += self._harvest_profit(date, delta)
        return r




from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from cyclesgym.managers import *
from cyclesgym.managers.utils import *

from cyclesgym.paths import TEST_PATH


class TestOperationManager(unittest.TestCase):
    def setUp(self):
        self.fname = TEST_PATH.joinpath('DummyOperation.operation')
        self.manager = OperationManager(self.fname)
        self.target = {
            (1, 106, 'FIXED_FERTILIZATION'): {'MASS': 150., 'LAYER': 1.,
                                              'N_NH4': 0.75, 'N_NO3': 0.25},
            (1, 106, 'PLANTING'): {'CROP': 'CornRM.90', 'FRACTION': 1.},
            (1, 106, 'TILLAGE'): {'DEPTH': 0.03, 'SOIL_DISTURB_RATIO': 5.},
        }

    def test_parse(self):
        assert self.target.keys() == self.manager.op_dict.keys()
        for k in self.target.keys():
            v_target = dict(sorted(self.target[k].items()))
            v_actual = dict(sorted(self.manager.op_dict[k].items()))

            assert v_target == v_actual

    def test_to_str(self):
        assert compare_stripped_string(read_skip_comments(self.fname),
                                       self.manager._to_str())


class TestCropManager(unittest.TestCase):
    def setUp(self):
        self.fname = TEST_PATH.joinpath('DummyCrop.dat')
        self.manager = CropManager(self.fname)
        self.target = pd.DataFrame({'YEAR': pd.Series([1980, 1980], dtype=int),
                               'DOY': pd.Series([1, 2], dtype=int),
                               'CROP': pd.Series(['FALLOW', 'FALLOW'], dtype=str),
                               'STAGE': pd.Series(['N/A', 'N/A'], dtype=str),
                               'THERMAL TIME': pd.Series([0, 0], dtype=float),
                               'CUM. BIOMASS': pd.Series([1, 0], dtype=float),
                               'AG BIOMASS': pd.Series([2, 0], dtype=float),
                               'ROOT BIOMASS': pd.Series([3, 0], dtype=float),
                               'FRAC INTERCEP': pd.Series([4, 0], dtype=float),
                               'TOTAL N': pd.Series([5, 0], dtype=float),
                               'AG N': pd.Series([6, 0], dtype=float),
                               'ROOT N': pd.Series([7, 0], dtype=float),
                               'AG N CONCN': pd.Series([8, 0], dtype=float),
                               'N FIXATION': pd.Series([9, 0], dtype=float),
                               'N ADDED': pd.Series([10, 0], dtype=float),
                               'N STRESS': pd.Series([11, 0], dtype=float),
                               'WATER STRESS': pd.Series([12, 0], dtype=float),
                               'POTENTIAL TR': pd.Series([13, 1.5], dtype=float)})

    def test_parse(self):
        assert self.target.equals(self.manager.crop_state)

    def test_get_day(self):
        assert self.manager.get_day(year=1980, doy=2).equals(self.target.iloc[1:2, :])

    def test_to_str(self):
        s = read_skip_comments(self.fname).split('\n')[0]
        old_crop_state = self.manager.crop_state.copy()

        # Make sure writing did not affect the original df
        assert old_crop_state.equals(self.manager.crop_state)
        assert compare_stripped_string(read_skip_comments(self.fname),
                                       self.manager._to_str())


class TestWeatherManager(unittest.TestCase):
    def setUp(self):
        self.fname = TEST_PATH.joinpath('DummyWeather.weather')
        self.manager = WeatherManager(self.fname)
        self.target_immutables = pd.DataFrame({'LATITUDE': [40.687500],'ALTITUDE': [0.0], 'SCREENING_HEIGHT': [10.0]})
        self.target_mutables = pd.DataFrame({'YEAR': pd.Series([1980, 1980], dtype=int),
                                             'DOY': pd.Series([1, 2], dtype=int),
                                             'PP': pd.Series([0, 0], dtype=float),
                                             'TX': pd.Series([1, 0], dtype=float),
                                             'TN': pd.Series([2, 0], dtype=float),
                                             'SOLAR': pd.Series([3, 0], dtype=float),
                                             'RHX': pd.Series([4, 0], dtype=float),
                                             'RHN': pd.Series([5, 0], dtype=float),
                                             'WIND': pd.Series([6, 1.5], dtype=float),})

    def test_parse(self):
        assert self.manager.immutables.equals(self.target_immutables)
        assert self.manager.mutables.equals(self.target_mutables)

    def test_to_str(self):
        # print(read_skip_comments(self.fname))
        # print(self.manager._to_str())
        assert compare_stripped_string(read_skip_comments(self.fname),
                                       self.manager._to_str())

    def test_consistency(self):
        """
        Test that loading a saved file results in same dataframes.
        """
        # TODO: Add same test for other managers
        fname = Path(__file__).parent.joinpath(
            'DummyWeatherFromManager.weather')
        self.manager.save(fname)
        load_manager = WeatherManager(fname)
        assert self.manager.immutables.equals(load_manager.immutables)
        assert self.manager.mutables.equals(load_manager.mutables)




class TestControlManager(unittest.TestCase):
    def setUp(self):
        self.fname = TEST_PATH.joinpath('DummyControl.ctrl')
        self.manager = ControlManager(self.fname)
        self.target ={'SIMULATION_START_YEAR'   :1980,
                      'SIMULATION_END_YEAR'     :1980,
                      'ROTATION_SIZE'           :1,
                      'USE_REINITIALIZATION'    :0,
                      'ADJUSTED_YIELDS'         :0,
                      'HOURLY_INFILTRATION'     :1,
                      'AUTOMATIC_NITROGEN'      :0,
                      'AUTOMATIC_PHOSPHORUS'    :0,
                      'AUTOMATIC_SULFUR'        :0,
                      'DAILY_WEATHER_OUT'       :1,
                      'DAILY_CROP_OUT'          :1,
                      'DAILY_RESIDUE_OUT'       :1,
                      'DAILY_WATER_OUT'         :1,
                      'DAILY_NITROGEN_OUT'      :1,
                      'DAILY_SOIL_CARBON_OUT'   :1,
                      'DAILY_SOIL_LYR_CN_OUT'   :1,
                      'ANNUAL_SOIL_OUT'         :1,
                      'ANNUAL_PROFILE_OUT'      :1,
                      'ANNUAL_NFLUX_OUT'        :1,
                      'CROP_FILE'               :'GenericCrops.crop',
                      'OPERATION_FILE'          :'ContinuousCorn.operation',
                      'SOIL_FILE'               :'GenericHagerstown.soil',
                      'WEATHER_FILE'            :'RockSprings.weather',
                      'REINIT_FILE'             :'N/A'}

    def test_parse(self):
        assert self.manager.ctrl_dict == self.target

    def test_to_str(self):
        assert compare_stripped_string(read_skip_comments(self.fname),
                                       self.manager._to_str())


class TestSeasonManager(unittest.TestCase):
    def setUp(self):
        self.fname = TEST_PATH.joinpath('DummySeason.dat')
        self.manager = SeasonManager(self.fname)
        columns = ['YEAR', 'DOY', 'CROP', 'PLANT_YEAR', 'PLANT_DOY', 'TOTAL BIOMASS',
                   'ROOT BIOMASS', 'GRAIN YIELD', 'FORAGE YIELD', 'AG RESIDUE',
                   'HARVEST INDEX', 'POTENTIAL TR', 'ACTUAL TR', 'SOIL EVAP',
                   'IRRIGATION', 'TOTAL N', 'ROOT N', 'GRAIN N', 'FORAGE N',
                   'CUM. N STRESS', 'N IN HARVEST', 'N IN RESIDUE',
                   'N CONCN FORAGE', 'N FIXATION', 'N ADDED']
        values = [[1980, 252, 'CornRM.90', 1980, 110] +
                  list(np.arange(1, 21).astype(float))]
        self.target = pd.DataFrame(data=values, index=None, columns=columns)

    def test_parse(self):
        assert self.target.equals(self.manager.season_df)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.date_df = pd.DataFrame({
            'DATE': ['1980-01-01', '1980-01-02'],
            'CROP': ['wheat', 'corn'],
            'PLANT_DATE': ['1981-01-01', '1981-01-02']
        })
        self.ydoy_df = pd.DataFrame({
            'YEAR': [1980, 1980],
            'DOY': [1, 2],
            'CROP': ['wheat', 'corn'],
            'PLANT_YEAR': [1981, 1981],
            'PLANT_DOY': [1, 2]
        })

    def test_datetoydoy(self):
        new_df = date_to_ydoy(self.date_df, 'DATE',
                              new_col_names=['YEAR', 'DOY'], inplace=False)
        new_df = date_to_ydoy(new_df, 'PLANT_DATE',
                              new_col_names=['PLANT_YEAR', 'PLANT_DOY'],
                              inplace=False)
        assert not new_df.equals(self.date_df)
        assert new_df.equals(self.ydoy_df)

    def test_datetoydoy_inplace(self):
        new_df = date_to_ydoy(self.date_df, 'DATE',
                              new_col_names=['YEAR', 'DOY'], inplace=True)
        new_df = date_to_ydoy(new_df, 'PLANT_DATE',
                              new_col_names=['PLANT_YEAR', 'PLANT_DOY'],
                              inplace=True)
        assert new_df.equals(self.date_df)
        assert new_df.equals(self.ydoy_df)

    def test_ydoytodate(self):
        new_df = ydoy_to_date(self.ydoy_df, old_col_names=['YEAR', 'DOY'],
                             new_col_name='DATE', inplace=False)
        new_df = ydoy_to_date(new_df, old_col_names=['PLANT_YEAR', 'PLANT_DOY'],
                             new_col_name='PLANT_DATE', inplace=False)
        assert not new_df.equals(self.ydoy_df)
        assert new_df.equals(self.date_df)

    def test_ydoytodate_inplace(self):
        new_df = ydoy_to_date(self.ydoy_df, old_col_names=['YEAR', 'DOY'],
                              new_col_name='DATE', inplace=True)
        new_df = ydoy_to_date(new_df,
                              old_col_names=['PLANT_YEAR', 'PLANT_DOY'],
                              new_col_name='PLANT_DATE', inplace=True)
        assert new_df.equals(self.ydoy_df)
        assert new_df.equals(self.date_df)


def compare_stripped_string(s1, s2):
    """
    Compared two strings line by line removing white spaces.
    """
    l1 = list(filter(None, s1.split('\n')))
    l2 = list(filter(None, s2.split('\n')))

    if len(l1) != len(l2):
        return False
    else:
        equal = True
        for el1, el2 in zip(l1, l2):
            equal &= el1.replace(' ', '') == el2.replace(' ', '')
            if not equal:
                print(el1)
                print(el2)
    return equal


def read_skip_comments(fname):
    """
    Read a file removing comments.
    """
    s = ''
    with open(fname, 'r') as fp:
        for line in fp:
            line = line.strip(' ')
            if line.startswith('#'):
                continue
            s += line
    return s


if __name__ == '__main__':
    unittest.main()
from datetime import timedelta
from cyclesgym.envs.corn import CornNew
from typing import Tuple
import shutil
import numpy as np
import pathlib

__all__ = ['CornMultiYear']


class _CornMultiYearContinue(CornNew):

    def __init__(self, START_YEAR, END_YEAR, delta, n_actions, maxN):
        self.ROTATION_SIZE = END_YEAR - START_YEAR + 1
        super(CornNew, self).__init__(SIMULATION_START_YEAR=START_YEAR,
                                      SIMULATION_END_YEAR=END_YEAR,
                                      ROTATION_SIZE=self.ROTATION_SIZE,
                                      USE_REINITIALIZATION=0,
                                      ADJUSTED_YIELDS=0,
                                      HOURLY_INFILTRATION=1,
                                      AUTOMATIC_NITROGEN=0,
                                      AUTOMATIC_PHOSPHORUS=0,
                                      AUTOMATIC_SULFUR=0,
                                      DAILY_WEATHER_OUT=0,
                                      DAILY_CROP_OUT=1,
                                      DAILY_RESIDUE_OUT=0,
                                      DAILY_WATER_OUT=0,
                                      DAILY_NITROGEN_OUT=0,
                                      DAILY_SOIL_CARBON_OUT=0,
                                      DAILY_SOIL_LYR_CN_OUT=0,
                                      ANNUAL_SOIL_OUT=0,
                                      ANNUAL_PROFILE_OUT=0,
                                      ANNUAL_NFLUX_OUT=0,
                                      CROP_FILE='GenericCrops.crop',
                                      OPERATION_FILE='ContinuousCorn.operation',
                                      SOIL_FILE='GenericHagerstown.soil',
                                      WEATHER_FILE='RockSprings.weather',
                                      REINIT_FILE='N / A',
                                      delta=delta)
        self._post_init_setup()
        self._generate_observation_space()
        self._generate_action_space(n_actions, maxN)

    def _create_operation_file(self):
        """Create operation file by copying the base one."""
        super(_CornMultiYearContinue, self)._create_operation_file()
        operations = [key for key in self.op_manager.op_dict.keys() if key[2] != 'FIXED_FERTILIZATION']
        for i in range(self.ROTATION_SIZE-1):
            for op in operations:
                copied_op = (i+2,) + op[1:]
                self.op_manager.op_dict[copied_op] = self.op_manager.op_dict[op]

        self.op_manager.save(self.op_file)
        #TODO: write other operations only if not available. Print warining in this case.

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, r, done, _ = super(_CornMultiYearContinue, self).step(action)
        return obs, r, done, self.date


class CornMultiYear(_CornMultiYearContinue):

    def _create_control_file(self):
        super(CornMultiYear, self)._create_control_file()
        self.reinit_year = int((self.ctrl_base_manager.ctrl_dict['SIMULATION_START_YEAR']
                                + self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR'])/2)
        self.ctrl_manager.ctrl_dict['SIMULATION_END_YEAR'] = self.reinit_year
        self.ctrl_manager.save(self.ctrl_file)

    def _update_reinit_file(self):
        new_reinit = self.input_dir.name.joinpath('reinit.dat')
        shutil.copy(self._get_output_dir().joinpath('reinit.dat'), new_reinit)

        lines = []
        with open(new_reinit, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line_splitted = line.split()
                if len(line_splitted) > 0:
                    if line_splitted[1].isnumeric():
                        if int(line_splitted[1]) == self.reinit_year:
                            indx = i
                            line = line_splitted
                            line[1] = str(int(line[1]) + 1)
                            line[3] = '1'
                            line = line[0] + '    ' + line[1] + '    ' + line[2] + '     ' + line[3] + '\n'
                lines.append(line)

        with open(new_reinit, 'w') as f:
            for i, line in enumerate(lines):
                if i >= indx:
                    f.write(line)

    def _update_control_file(self):
        self.ctrl_manager.ctrl_dict['SIMULATION_START_YEAR'] = self.reinit_year + 1
        self.ctrl_manager.ctrl_dict['SIMULATION_END_YEAR'] = self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR']
        self.ctrl_manager.ctrl_dict['USE_REINITIALIZATION'] = 1
        self.ctrl_manager.ctrl_dict['REINIT_FILE'] = pathlib.Path(self.input_dir.name.stem).joinpath('reinit.dat')
        self.ctrl_manager.save(self.ctrl_file)

    def _update_operation_file(self):
        reference_year = self.reinit_year-self.ctrl_base_manager.ctrl_dict['SIMULATION_START_YEAR'] + 1
        for key in list(self.op_manager.op_dict.keys()):
            operation = self.op_manager.op_dict.pop(key)
            if key[0] > reference_year:
                self.op_manager.op_dict[(key[0] - reference_year, *key[1:])] = operation

        self.op_manager.save(self.op_file)

    def _check_is_mid_year(self):
        year_of_next_step = (self.date + timedelta(days=self.delta)).year
        return (year_of_next_step > self.reinit_year and self.date.year == self.reinit_year)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        N_mass = self._action2mass(action)
        rerun_cycles = self.implementer.implement_action(
            date=self.date, mass=N_mass)

        reinit = self._check_is_mid_year()
        doy = None
        if reinit:
            doy = 365

        if rerun_cycles or reinit:
            self._call_cycles(debug=False, reinit=reinit, doy=doy)

        # Advance time
        self.date += timedelta(days=self.delta)
        if reinit:
            self._update_control_file()
            self._update_reinit_file()
            self._update_operation_file()
            self.implementer.start_year = self.reinit_year + 1

        self.season_manager.update_file(self.season_file)
        # Compute reward
        r = self.rewarder.compute_reward(date=self.date, delta=self.delta, action=N_mass)

        done = self.date.year > self.ctrl_base_manager.ctrl_dict['SIMULATION_END_YEAR']

        if reinit and not done:
            self._call_cycles(debug=False)

        # Compute state
        obs = self.observer.compute_obs(self.date, N=N_mass)

        # Compute
        return obs, r, done, self.date


import numpy as np
import warnings
from pathlib import Path
import datetime
from cyclesgym.managers.operation import OperationManager
from cyclesgym.envs.utils import date2ydoy

__all__ = ['Fertilizer', 'FixedRateNFertilizer']


class Fertilizer(object):
    valid_nutrients = ['C_Organic', 'C_Charcoal', 'N_Organic', 'N_Charcoal',
                     'N_NH4', 'N_NO3', 'P_Organic', 'P_CHARCOAL',
                     'P_INORGANIC', 'K', 'S']

    def __init__(self,
                 operation_manager: OperationManager,
                 operation_fname: Path,
                 affected_nutrients: list,
                 start_year: int) -> None:
        self.operation_manager = operation_manager
        self.operation_fname = operation_fname
        assert all([nutrient in self.valid_nutrients for nutrient in
                    affected_nutrients]), f'Affected nutrients should be in ' \
                                          f'{self.valid_nutrients}. Got' \
                                          f'{affected_nutrients} instead.'
        self.affected_nutrients = affected_nutrients
        self.start_year = start_year

    def _is_new_action(self, year: int, doy: int, new_masses: dict) -> bool:
        """
        Is the fertilization recommended for (year, doy) different from previous one?

        Parameters
        ----------
        year: int
            Year
        doy: int
            Day of the year
        new_masses: dict
            keys are nutrients and values are masses in Kg

        Returns
        -------
        new: bool
        """
        # Get old operation
        old_op = self.operation_manager.op_dict.get(
            (year, doy, 'FIXED_FERTILIZATION'))
        if old_op is None:
            old_masses = {n: 0 for n in self.affected_nutrients}
        else:
            old_masses = {n: v for n, v in old_op.items() if n in self.affected_nutrients}
        return old_masses != new_masses

    def year2opyear(self, year: int) -> int:
        """
        Convert normal year to years used in operation.

        Cycles and cyclesgym use normal years (e.g. 1980) but, in the operation
        file, they start counting from the first simulation year, i.e., if the
        sim start in 1980, this is indicated as year 1 in the operation file.
        """
        return year - self.start_year + 1

    def implement_action(self, date: datetime.date, masses: dict) -> bool:
        """
        Write new operation file.

        We check if the new action is different from the previously specified
        one. If it is, we overwrite the operation file and we indicate cycles
        should be rerun. Othwerwise, we do nothing.

        Parameters
        ----------
        date: datetime.date
            Date
        masses: dict
            keys are nutrients and values are masses in Kg.

        Returns
        -------
        rerun_cycles: bool
            if true, it means the operation file has been changed and cycles
            should be rerun.
        """
        # Check all and only affacted nutrients are specified
        assert all(n in self.affected_nutrients for n in masses.keys()), f'You can only specify fertilization masses for {self.affected_nutrients}'
        assert all(n in masses.keys() for n in self.affected_nutrients), f'You must specify fertilization masses for all {self.affected_nutrients}'

        year, doy = date2ydoy(date)
        year = self.year2opyear(year)

        if self._is_new_action(year, doy, masses):
            # Check for collision
            key = (year, doy, 'FIXED_FERTILIZATION')
            fertilization_op = self.operation_manager.op_dict.get(key)
            collision = fertilization_op is not None

            if collision:
                # Take existing operations and update masses
                op = {key: self._update_fertilization_op(fertilization_op,
                                                         masses,
                                                         mode='absolute')}
            else:
                # Initialize operation
                op_val = {
                    'SOURCE': 'Unknown',
                    'MASS': 0,
                    'FORM': 'Unknown',
                    'METHOD': 'Unknown',
                    'LAYER': 1,  # TODO: Handle different layers
                }

                # Initialize all nutrients to zero
                op_val.update({n: 0 for n in self.valid_nutrients})

                # Update value for affected nutrients
                total_mass = sum([m for m in masses.values()])
                op_val.update({'MASS': total_mass})
                op_val.update({nutrient: mass / total_mass for
                               (nutrient, mass) in masses.items()})
                op = {key: op_val}
            # Insert and write operation
            self.operation_manager.insert_new_operations(op, force=True)
            self.operation_manager.save(self.operation_fname)

            rerun_cycles = True
        else:
            # If not new action, don't update operation and don't rerun
            rerun_cycles = False
        return rerun_cycles

    def reset(self):
        """
        Set to zero the masses for all the affected nutrients.

        Before starting a new simulation, we want to make sure that the only
        way the affected nutrients can be provided is through our
        fertilization. This is why we reset everything to zero (overwriting
        during the unrolling of the env may not be sufficient as there may be
        fertilization event happening between t and t + delta that we could not
        overwrite.
        """
        for op_k, op_v in self.operation_manager.op_dict.items():
            if op_k[-1] == 'FIXED_FERTILIZATION':
                new_op = self._update_fertilization_op(
                    old_op=op_v,
                    new_masses={n: 0 for n in self.affected_nutrients},
                    mode='absolute')
                self.operation_manager.insert_new_operations(
                    {op_k: new_op}, force=True)
        self.operation_manager.save(self.operation_fname)

    def _update_fertilization_op(self, old_op: dict, new_masses: dict,
                                 mode: str = 'absolute') -> dict:
        """
        Update the masses of a fertilization operation.

        Parameters
        ----------
        old_op: dict
            Dictionary of fertilization operation to update
        new_masses: dict
            Dict of new masses. Keys should be possible nutrients
        mode: str
            With 'increment' the mass is added to the one already present
            in the operation. With 'absolute' we use the new mass regardless of
            what was already there
        Returns
        -------
        new_op: dict
            Dictionary of updated fertilization operation
        """

        assert mode in ['increment', 'absolute'], \
            'Can only update in increment or absolute mode'

        assert all(k in self.valid_nutrients for k in new_masses.keys()), \
            f'New masses can only specify valid nutrients ' \
            f'({self.valid_nutrients}) as keys'

        # Copy operation
        new_op = old_op.copy()

        # Init final masses
        final_masses = {k: 0 for k in self.valid_nutrients}

        for i, n in enumerate(self.valid_nutrients):
            # Update with new masses when specified
            if n in new_masses.keys():
                final_masses[n] = new_masses[n] if mode == 'absolute' else \
                    new_masses[n] + old_op[n]
            # Copy old masses otherwise
            else:
                final_masses[n] = old_op[n] * old_op['MASS']

        # Update dict
        total_mass = sum(final_masses.values())
        new_op['MASS'] = total_mass
        if total_mass > 0:
            new_op.update(
                {nutrient: final_mass / total_mass for (nutrient, final_mass)
                 in final_masses.items()})
        else:
            new_op.update({nutrient: 0 for nutrient in self.valid_nutrients})

        return new_op


class FixedRateNFertilizer(Fertilizer):
    def __init__(self, operation_manager: OperationManager,
                 operation_fname: Path,
                 rate: float,
                 start_year: int):
        nutrients = ['N_NH4', 'N_NO3']
        super().__init__(operation_manager,
                         operation_fname,
                         nutrients,
                         start_year)
        assert 0 <= rate <= 1, f'Rate must be in [0, 1]. It is {rate} instead'
        self.rate = rate

    def convert_mass(self, mass):
        masses = {'N_NH4': mass * self.rate,
                  'N_NO3': mass * (1 - self.rate)}
        return masses

    # TODO: This should have same signature as parent method!
    def implement_action(self, date: datetime.date, mass: float):
        masses = self.convert_mass(mass)
        return super().implement_action(date, masses)


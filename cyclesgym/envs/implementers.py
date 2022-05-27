import numpy as np
import warnings
from pathlib import Path
import datetime
from cyclesgym.managers.operation import OperationManager
from cyclesgym.envs.utils import date2ydoy

__all__ = ['Fertilizer', 'FixedRateNFertilizer']


class Implementer(object):

    def year2opyear(self, year: int) -> int:
        """
        Convert normal year to years used in operation.

        Cycles and cyclesgym use normal years (e.g. 1980) but, in the operation
        file, they start counting from the first simulation year, i.e., if the
        sim start in 1980, this is indicated as year 1 in the operation file.
        """
        return year - self.start_year + 1

    def _is_new_action(self, year: int, doy: int, *args, **kwargs) -> bool:
        raise NotImplementedError

    def reset(self) -> bool:
        raise NotImplementedError

    def _check_valid_action(self, action_details: dict):
        raise NotImplementedError

    def _update_operation(self, year, doy, operation, *args, **kwargs):
        raise NotImplementedError

    def _check_collision(self, year, doy, operation_details):
        raise NotImplementedError

    def _get_operation_key(self, year, doy, operation_details):
        raise NotImplementedError

    def _create_new_operation(self, year, key, operation_details):
        raise NotImplementedError

    def implement_action(self, date: datetime.date, operation_details: dict) -> bool:
        """
        Write new operation file.

        We check if the new action is different from the previously specified
        one. If it is, we overwrite the operation file and we indicate cycles
        should be rerun. Othwerwise, we do nothing.

        Parameters
        ----------
        date: datetime.date
            Date
        operation_details: dict
                    details of the operation to be implemented

        Returns
        -------
        rerun_cycles: bool
            if true, it means the operation file has been changed and cycles
            should be rerun.
        """
        # Check all and only affected crops are specified
        self._check_valid_action(operation_details)

        year, doy = date2ydoy(date)
        year = self.year2opyear(year)

        if self._is_new_action(year, doy, operation_details):
            # Check for collision
            operation, collision = self._check_collision(year, doy, operation_details)
            if collision:
                # Take existing operations and update masses
                op = self._update_operation(year, doy, operation, operation_details)
            else:
                op = self._create_new_operation(year, doy, operation_details)

            # Insert and write operation
            self.operation_manager.insert_new_operations(op, force=True)
            self.operation_manager.save(self.operation_fname)

            rerun_cycles = True
        else:
            # If not new action, don't update operation and don't rerun
            rerun_cycles = False
        return rerun_cycles


class Fertilizer(Implementer):
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
            total_mass = old_op['MASS']
            old_masses = {n: v * total_mass for n, v in old_op.items()
                          if n in self.affected_nutrients}
        return old_masses != new_masses

    def _get_operation_key(self, year, doy):
        return (year, doy, 'FIXED_FERTILIZATION')

    def _check_valid_action(self, action_details: dict):
        assert all(n in self.affected_nutrients for n in
                   action_details.keys()), f'You can only specify fertilization masses for {self.affected_nutrients}'
        assert all(n in action_details.keys() for n in
                   self.affected_nutrients), f'You must specify fertilization masses for all {self.affected_nutrients}'

    def _create_new_operation(self, year, doy, masses: dict) -> dict:
        key = self._get_operation_key(year, doy)
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
        return op

    def reset(self) -> bool:
        """
        Set to zero the masses for all the affected nutrients.

        Before starting a new simulation, we want to make sure that the only
        way the affected nutrients can be provided is through our
        fertilization. This is why we reset everything to zero (overwriting
        during the unrolling of the env may not be sufficient as there may be
        fertilization event happening between t and t + delta that we could not
        overwrite.
        """
        doy = 1
        rerun_cycles = False

        for op_k, op_v in self.operation_manager.op_dict.items():
            if op_k[-1] == 'FIXED_FERTILIZATION':
                start_year = op_k[0]
                doy = op_k[1]
                new_op = self._update_operation(start_year, doy, 
                                                old_op = op_v,
                                                new_masses={n: 0 for n in self.affected_nutrients},
                                                mode='absolute')
                self.operation_manager.insert_new_operations(
                    new_op, force=True)
                rerun_cycles = True
        self.operation_manager.save(self.operation_fname)
        return rerun_cycles

    def _check_collision(self, year, doy, operation_details):
        key = (year, doy, 'FERTILIZATION')
        operation = self.operation_manager.op_dict.get(key)
        collision = operation is not None
        return operation, collision

    def _update_operation(self, year, doy,
                          old_op: dict, new_masses: dict,
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

        # TODO: include mode in the call of the Abstract class in some way. Right now abstract class has no "mode"

        key = self._get_operation_key(year, doy) # TODO: Weird we get a dict with massese as input and return a dict {k: masses} as output. Can't we add the key later?

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

        return {key: new_op}


class FixedRateNFertilizer(Fertilizer):
    def __init__(self, operation_manager: OperationManager,
                 operation_fname: Path,
                 rate: float,
                 start_year: int):
        nutrients = ['N_NH4', 'N_NO3']
        super(FixedRateNFertilizer, self).__init__(operation_manager,
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
        return super(FixedRateNFertilizer, self).implement_action(date, masses)


class Planter(Implementer):
    valid_crops = ['CornRM.90', 'CornRM.100', 'CornRM.110', 'CornSilageRM.90', 'CornSilageRM.100', 'CornSilageRM.110',
                   'SorghumSS', 'SorghumMS', 'SorghumLS', 'SweetSorghum', 'SoybeanMG.3', 'SoybeanMG.4', 'SoybeanMG.5',
                   'OatsGrain', 'OatsHay', 'RygrassAnnualGrazing', 'SpringWheat', 'SpringBarley', 'WinterWheat',
                   'WinterBarley', 'Chickpea', 'SpringPeas', 'WinterPeas', 'SpringLentils', 'WinterLentils',
                   'SpringCanola', 'WinterCanola', 'Potatoes', 'Millet', 'Sesame_Beta', 'Teff_Beta', 'Alfalfa',
                   'LotusCorniculatus', 'WhiteClover', 'Orchardgrass', 'TallFescueGrazing', 'Switchgrass', 'Miscanthus',
                   'Willow', 'C3_weed', 'C4_weed']

    def __init__(self,
                 operation_manager: OperationManager,
                 operation_fname: Path,
                 affected_crops: list,
                 start_year: int) -> None:
        self.operation_manager = operation_manager
        self.operation_fname = operation_fname
        assert all([crop in self.valid_crops for crop in
                    affected_crops]), f'Affected crops should be in ' \
                                          f'{self.valid_crops}. Got' \
                                          f'{affected_crops} instead.'
        self.affected_crops = affected_crops
        self.start_year = start_year

    def _is_new_action(self, year: int, doy: int, crop_details: dict) -> bool:
        """
        Is the crop recommended for (year, doy) different from previous one?

        Parameters
        ----------
        year: int
            Year
        doy: int
            Day of the year
        crop: str
            crop to be planted

        Returns
        -------
        new: bool
        """
        # Get old operation
        old_op = self.operation_manager.op_dict.get(
            (year, doy, 'PLANTING'))
        if old_op is None:
            return True
        else:
            return old_op['CROP'] == crop_details['CROP']

    def _check_valid_action(self, action_details: dict):
        assert (action_details['CROP'] in self.valid_crops), f'You can only specify planting date for ' \
                                                             f'{self.valid_crops}'

    def reset(self) -> bool:
        """
        Set to zero the planting events of all the affected crops.

        Before starting a new simulation, we want to make sure that the only
        way the affected crops can be planted is using the planter implementer.
        This is why we delete all the operations that involve planting affected
        crops
        """
        rerun_cycles = True
        op_to_delete = []
        for op_k, op_v in self.operation_manager.op_dict.items():
            if op_k[-1] == 'PLANTING' and op_v['CROP'] in self.affected_crops:
                op_to_delete.append(op_k)

        self.operation_manager.delete_operations(op_to_delete)
        self.operation_manager.save(self.operation_fname)
        # TODO: Understand when we can avoid rerunning cycles
        return rerun_cycles

    def _get_operation_key(self, year, doy):
        return (year, doy, 'PLANTING')

    def _check_collision(self, year, doy, operation_details):
        planting_events = [key for key in self.operation_manager.op_dict.keys() if key[2] == 'PLANTING']
        operation = [key for key in planting_events if key[0] == year]
        collision = False
        if len(operation) > 0:
            operation = operation[0]
            operation = {operation: self.operation_manager.op_dict[operation]}
            collision = True
        return operation, collision

    def _update_operation(self, year, doy, old_operation, operation_details) -> dict:
        """
        Update the planting operation.

        Parameters
        ----------
        old_operation: dict
            Dictionary of operation detials of old op
        operation_details: dict
            Dict of new operation.
        -------
        new_op: dict
            Dictionary of updated planting operation
        """

        assert operation_details['CROP'] in self.valid_crops, \
            f'New planting operation can only specify valid crops \n' \
            f'({self.valid_crops}) as planted crop \n' \
            f"Given {operation_details['CROP']}"

        doy = operation_details.pop('DOY')
        for item in old_operation.values():
            item.update(operation_details)

        old_key = list(old_operation.keys())[0]
        new_key = (old_key[0], doy, old_key[2])
        old_operation[new_key] = old_operation.pop(old_key)

        self.operation_manager.op_dict.pop(old_key)
        return old_operation

    def _create_new_operation(self, year, doy, operation_details: dict) -> dict:
        key = self._get_operation_key(year, doy)
        op_val = {'DOY': 1,
                  'END_DOY': -999,
                  'MAX_SMC': -999,
                  'MIN_SMC': 0.0,
                  'MIN_SOIL_TEMP': 0.0,
                  'CROP': None,
                  'USE_AUTO_IRR': 0,
                  'USE_AUTO_FERT': 0,
                  'FRACTION': 1.0,
                  'CLIPPING_START': 1,
                  'CLIPPING_END': 366
                  }

        # Initialize all nutrients to zero
        op_val.update(operation_details)
        key = (key[0], op_val.pop('DOY'), key[2])
        return {key: op_val}


class RotationPlanter(Planter):
    def __init__(self, operation_manager: OperationManager,
                 operation_fname: Path,
                 rotation_crops: list,
                 start_year: int):
        super(RotationPlanter, self).__init__(operation_manager,
                                              operation_fname,
                                              rotation_crops,
                                              start_year)

    def convert_action_to_dict(self, crop_categorical: int, doy: int,
                               end_doy: int, max_smc: int):
        """crop_categorical: integer representing the crop in the crop rotation
        doy: integer lower bound of the planting date, as number of weeks starting from the
            first of march
        end_doy: upper bound of the planting date, as number of weeks from the doy
        max_smc: maximum moisture for automated planting"""

        doy = 90 + doy * 7
        end_doy = doy + end_doy * 7
        max_smc = float(max_smc / 10)

        operation_det = {'DOY': doy,
                         'CROP': self.affected_crops[crop_categorical],
                         'END_DOY': end_doy,
                         'MAX_SMC': max_smc}
        return operation_det

    def implement_action(self, date: datetime.date, crop_categorical: int, doy: int,
                         END_DOY: int, MAX_SMC: int):
        operation_det = self.convert_action_to_dict(crop_categorical, doy, END_DOY, MAX_SMC)
        return super(RotationPlanter, self).implement_action(date, operation_det)
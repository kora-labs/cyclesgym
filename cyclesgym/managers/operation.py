import warnings

from cyclesgym.managers.common import InputFileManager

__all__ = ['OperationManager']


class OperationManager(InputFileManager):
    OPERATION_TYPES = (
    'TILLAGE', 'PLANTING', 'FIXED_FERTILIZATION', 'FIXED_IRRIGATION',
    'AUTO_IRRIGATION')

    def __init__(self, fname=None):
        self.op_dict = dict()
        super().__init__(fname)

    def _valid_input_file(self, fname):
        if fname is None:
            return True
        else:
            return super(OperationManager, self)._valid_input_file(fname) and fname.suffix == '.operation'

    def _valid_output_file(self, fname):
        return super(OperationManager, self)._valid_output_file(fname) and fname.suffix == '.operation'

    def _parse(self, fname):
        if fname is not None:
            with open(fname, 'r') as f:
                line_n = None
                k = None
                self.op_dict = dict()
                # TODO: using (year, doy, operation_type) as key, creates conflicts when there is fertilization applied to different layers on the same day
                for line in f.readlines():
                    if line.startswith(self.OPERATION_TYPES):
                        operation = line.strip(' \n')
                        line_n = 0
                        if k is not None:
                            self.op_dict.update({k: v})
                    if line_n is not None:
                        if line_n == 1:
                            year = int(line.split()[1])
                        if line_n == 2:
                            doy = int(line.split()[1])
                            k = (year, doy, operation)
                            v = dict()
                        if line_n > 2:
                            if len(line.split()) > 0:
                                argument = line.split()[0]
                                value = line.split()[1]
                                try:
                                    value = float(value)
                                except ValueError:
                                    pass
                                v.update({argument: value})
                        line_n += 1
                if k is not None:
                    self.op_dict.update({k: v})
        else:
            self.op_dict = {}

    def _to_str(self):
        s = ''
        for k, v in self.op_dict.items():
            year, doy, operation = k
            s += operation + f"\n{'YEAR':30}\t{year}\n{'DOY':30}\t{doy}\n"
            for argument, value in v.items():
                if argument == 'operation':
                    pass
                else:
                    s += f'{argument:30}\t{value}\n'
            s += '\n'
        return s

    def save(self, fname, force=True):
        self.sort_operation()
        super().save(fname, force)

    def count_same_day_events(self, year, doy):
        return len(self.get_same_day_events(year, doy))

    def get_same_day_events(self, year, doy):
        return {k: v for k, v in self.op_dict.items() if k[0] == year and k[1] == doy}

    def sort_operation(self):
        self.op_dict = dict(sorted(self.op_dict.items()))

    def _insert_single_operation(self, op_key, op_val, force=True):
        year, doy, operation = op_key
        assert operation in self.OPERATION_TYPES, \
            f'Operation must be one of the following {self.OPERATION_TYPES}'
        collisions = [operation == k[2] for k in self.get_same_day_events(year, doy).keys()]
        if any(collisions):
            warnings.warn(f'There is already an operation {operation} for they day {doy} of year {year}.')
            if force:
                self.op_dict.update({op_key: op_val})
            else:
                pass
        else:
            self.op_dict.update({op_key: op_val})

    def insert_new_operations(self, op, force=True):
        for k, v in op.items():
            self._insert_single_operation(k, v, force=force)
        self.sort_operation()

    def _delete_single_operation(self, k):
        self.op_dict.pop(k)

    def delete_operations(self, keys):
        for k in keys:
            self._delete_single_operation(k)
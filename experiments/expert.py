"""
Functions that output Fixed policies (action sequences)
"""

import pathlib
import warnings
import shutil
import numpy as np
import sys
import time

def create_action_sequence(doy, weight, maxN, n_actions, delta_t, n_weeks=52):
    doy = np.atleast_1d(doy)
    weight = np.atleast_1d(weight)
    assert len(doy) == len(weight)
    delta_a = maxN / (n_actions - 1)
    action_sequence = np.zeros(n_weeks, dtype=int)
    for d, w in zip(doy, weight):
        ind = np.floor(d / delta_t).astype(int)
        a = np.floor(w / delta_a).astype(int)
        action_sequence[ind] = a
    return action_sequence



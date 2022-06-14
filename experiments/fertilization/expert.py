"""
Functions that output Fixed policies (action sequences)
"""
import numpy as np


def create_action_sequence(doy, weight, maxN, n_actions, delta_t, n_weeks=53):
    doy = np.atleast_1d(doy)
    weight = np.atleast_1d(weight)
    assert len(doy) == len(weight)
    delta_a = maxN / (n_actions - 1)
    action_sequence = np.zeros(n_weeks, dtype=int)
    for d, w in zip(doy, weight):
        #puts planting at the start of the week from the date selected
        ind = np.floor(d / delta_t).astype(int)
        a = np.floor(w / delta_a).astype(int)
        action_sequence[ind] = a
    return action_sequence



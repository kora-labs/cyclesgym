eps = 1e-8


def maximum_absolute_percentage_error(y1, y2):
    return np.max(np.abs(y1-y2)/np.abs(y1+eps))


def mean_absolute_percentage_error(y1, y2):
    return np.mean(np.abs(y1-y2)/np.abs(y1+eps))

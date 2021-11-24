import numpy as np


def to_discrete_single(values, threshold):
    def _to_discrete(x, threshold):
        if np.isnan(x):
            return -1
        if x < threshold:
            return 0
        return 1

    fun = np.vectorize(_to_discrete)
    return fun(values, threshold)


def to_discrete_double(values, threshold_lo=0.01, threshold_hi=0.01, classes = None):
    if not classes:
        classes = [0,1,2]
    def _to_discrete(x, threshold_lo, threshold_hi):
        if np.isnan(x):
            return -1
        if x <= threshold_lo:
            return classes[0]
        elif threshold_lo < x < threshold_hi:
            return classes[1]
        else:
            return classes[2]
    fun = np.vectorize(_to_discrete)
    return fun(values, threshold_lo, threshold_hi)
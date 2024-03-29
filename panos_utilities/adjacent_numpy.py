import numpy as np
from collections import namedtuple


def different_sign(a, b):
    """
    Returns true if a and b have different signs, or if a is zero an b is
    nonzero. Element-wise
    calculation for numpy arrays.
    :param a: number of numpy array
    :param b: number of numpy array
    :return: boolean or numpy.array(dtype=bool)
    """
    return ((a <= 0) & (b > 0)) | ((a >= 0) & (b < 0))


def zero_cross_boolean_index(a):
    """
    :param a: numpy array
    :return: array of boolean indices, indicating the first of two elements
    adjacent elements with different sign.
    An element equal to zero is indicated only if followed by a nonzero one.
    e.g.

    zero_cross_boolean_index(np.array([-1,0,1]))
    array([False,  True, False])


    zero_cross_boolean_index(np.array([0,0,1]))
    array([False,  True, False])

    zero_cross_boolean_index(np.array([]))
    array([], dtype=bool)
    """
    if a.size == 0:
        return np.array([], dtype=bool)

    did_it_cross = different_sign(a[0:-1], a[1:])

    return np.append(did_it_cross, [False])


Zero_Cross_Intervals = namedtuple('Zero_Cross_Intervals',
                                  ['x_before',
                                   'x_after',
                                   'y_before',
                                   'y_after'])


def zero_cross_elems(x, y):
    """
    find  the values of x for which y crosses zero. The values returned are the
    ones before or exactly at
    the zero crossing.If the last value of x is a root, it does not get
    returned
    :param x: numpy array (sorted)
    :param y: numpy array must have the same length with x
    :return: namedtuple Zero_Cross_Result (xbefore,xafter,ybefore,yafter) .
    """
    assert (x.size == y.size), "x and y must have equal size!"
    indices = np.arange(x.size)[zero_cross_boolean_index(y)]

    return Zero_Cross_Intervals(x[indices],
                                x[indices + 1],
                                y[indices],
                                y[indices + 1])

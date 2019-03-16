import numpy as np


def wrap_2pi(x):
    """
    maps number periodically on the range [0,2*pi)
    :param x: scalar or type convertible to numpy.array
    :return: scalar or numpy.array
    eg.

    >>> wrap_2pi([0, np.pi, 2 * np.pi])
    array([0.        , 3.14159265, 0.        ])

    >>> wrap_2pi(7)
    0.7168146928204138

    >>> wrap_2pi([3,4,5,6,7,8,9])
    array([3.        , 4.        , 5.        , 6.        , 0.71681469,
       1.71681469, 2.71681469])

    """
    return x - 2 * np.pi * np.floor_divide(x, 2 * np.pi)


def wrap_minus_pi_pi(x):
    """
    maps number periodically on the range [-pi,pi)
    :param x: scalar or type convertible to numpy.array
    :return: scalar or numpy.array

    >>> wrap_minus_pi_pi([0, np.pi, 2 * np.pi])
    array([ 0.        , -3.14159265,  0.        ])

    """
    return wrap_2pi(x + np.full_like(x, np.pi)) - np.pi

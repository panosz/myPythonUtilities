from . import adjacent_numpy as adn
from collections import namedtuple
import numpy as np
import scipy.optimize as optimize

version = '0.2'


def sample_fun(f, window, n_samples):
    '''
    samples a function uniformly for x in the range [window[0], window[1]].
    :param f: callable with a single float argument
    :param window: a tuple specifying the sampling range
    :param n_samples: int, the number of samples
    :return: a tuple of np.arrays x,y
    '''
    x0, x1 = window
    x = np.linspace(x0, x1, n_samples)
    return x, f(x)


def find_root(f, window, n_samples=100):
    '''
    find the roots of a scalar function within an specified interval.
    :param f: callable with float or numpy.array input
    :param window: tuple (xmin,xmax) defining the interval in which to search for roots
    :param n_samples: int, optional. number of samples distributed uniformly in window
    :return: sol: list of scipy.optimize.RootResults
    '''
    x, y = sample_fun(f, window, n_samples)

    cross = adn.zero_cross_elems(x, y)

    return [optimize.root_scalar(f, bracket=bra_ket) for bra_ket in zip(cross.x_before, cross.x_after)]


Level_Cross = namedtuple('LevelCross', ['level', 'solutions'])


def find_iso_points(f, iso_level, *args, **kwargs):
    """
    find the points where a scalar function crosses a specified horizontal line within a specified interval.
    :param f: callable with float or numpy.array input
    :param iso_level: float, horizontal level
    :param args: forwarded to find_root
    :param kwargs: forwarded to find_root
    :return: Level_Cross: a namedtuple with members
        level = float the iso level that is crossed and
        solutions = list of scipy.optimize.RootResults
    """

    def f_moved(*args_inner, **kwargs_inner):
        return f(*args_inner, **kwargs_inner) - iso_level

    root_list = find_root(f_moved, *args, **kwargs)

    return Level_Cross(iso_level, root_list)

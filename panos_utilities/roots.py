from . import adjacent_numpy as adn
import numpy as np
import scipy.optimize as optimize


version = '0.1'


def find_root(f, window, n_samples=100):
    """
    find the roots of a scalar function within an specified interval.
    :param f: callable with float or numpy.array input
    :param window: tuple (xmin,xmax) defining the interval in which to search for roots
    :param n_samples: int, optional. number of samples distributed uniformly in window
    :return: sol: list of scipy.optimize.RootResults
    """
    x0, x1 = window
    x = np.linspace(x0, x1, n_samples)
    y = f(x)
    x_before, x_after, _, _ = adn.zero_cross_elems(x, y)

    return [optimize.root_scalar(f, bracket=bra_ket) for bra_ket in zip(x_before, x_after)]


def find_iso_points(f, iso_level, *args, **kwargs):
    """
    find the points where a scalar function crosses a specified horizontal line within a specified interval.
    :param f: callable with float or numpy.array input
    :param iso_level: float, horizontal level
    :param args: forwarded to find_root
    :param kwargs: forwarded to find_root
    :return: sol: list of scipy.optimize.RootResults
    """
    def f_moved(*args_inner, **kwargs_inner):
        return f(*args_inner, **kwargs_inner) - iso_level

    return find_root(f_moved, *args, **kwargs)



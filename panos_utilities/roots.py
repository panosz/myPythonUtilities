from . import adjacent_numpy as adn
from . import adjacent as adj
from collections import namedtuple
import numpy as np
import scipy.optimize as optimize

__version__ = '0.2.0a0'


class RootConvergenceError(Exception):
    """
    Exception Raised for root that did not converge.

    Attributes:
        root -- root details
        message -- explanation of the error
    """

    def __init__(self, root, message=""):
        self.root = root
        self.message = message


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


def resolve_cross_intervals(f, iso_level, intervals: adn.Zero_Cross_Intervals):
    """
    find the solution f == iso_level. Each solution must be contained in an interval specified in the parameter intervals
    :param iso_level: scalar. specifies the rhs of the equation to solve: f==iso_level
    :type intervals: adn.Zero_Cross_Intervals
    :param f: callable
    :param intervals: adn.Zero_Cross_Intervals bracketing the solutions of f
    :return: a list of approximate solutions, one for each interval in intervals.
    """

    if iso_level:
        def f_moved(*args_inner, **kwargs_inner):
            return f(*args_inner, **kwargs_inner) - iso_level

        return [optimize.root_scalar(f_moved, bracket=bra_ket) for bra_ket in
                zip(intervals.x_before, intervals.x_after)]
    else:
        return [optimize.root_scalar(f, bracket=bra_ket) for bra_ket in zip(intervals.x_before, intervals.x_after)]


Level_Cross = namedtuple('LevelCross', ['level', 'solutions'])


def parse_root_result(root_result: optimize.zeros.RootResults):
    """
    looks into the rootResult and returns the root, if it has converged.
    Raises exception, if it has not
    :param root_result: optimize.zeros.RootResults
    """
    if root_result.converged:
        return root_result.root
    else:
        raise RootConvergenceError(root_result, "Root Did Not Converge")


def _find_iso_points_single_level(f, iso_level, window, n_samples):
    """
    find the points where a scalar function crosses a specified horizontal line within a specified interval.
    :param f: callable with float or numpy.array input
    :param iso_level: float, horizontal level
    :param window: tuple (xmin,xmax) defining the interval in which to search for roots
    :param n_samples: int, optional. number of samples distributed uniformly in window
    :return: Level_Cross: a namedtuple with members
        level = float the iso level that is crossed and
        solutions = list of scipy.optimize.RootResults
    """

    x, y = sample_fun(f, window, n_samples)

    cross = adn.zero_cross_elems(x, y - iso_level)
    root_list = resolve_cross_intervals(f, iso_level, cross)
    return Level_Cross(iso_level, root_list)



def _find_roots_single_level(f, iso_level, window, n_samples):
    """
    find the points where a scalar function crosses a specified horizontal line within a specified interval.
    :param f: callable with float or numpy.array input
    :param iso_level: float, horizontal level
    :param window: tuple (xmin,xmax) defining the interval in which to search for roots
    :param n_samples: int, optional. number of samples distributed uniformly in window
    :return: Level_Cross: a namedtuple with members
        level = float the iso level that is crossed and
        solutions = list of scalars

    Raises RootConvergenceError if a solution fails to converge
    """

    level, root_results = _find_iso_points_single_level(f,iso_level,window,n_samples)
    roots = [parse_root_result(root_result) for root_result in root_results]
    return Level_Cross(level,roots)


def find_iso_points(f, iso_levels, window, n_samples=100):
    """
    find the points where a scalar function crosses a specified horizontal line within a specified interval.
    :param f: callable with float or numpy.array input
    :param iso_levels: scalar or array_like: horizontal level for which to solve
    :param window: tuple (xmin,xmax) defining the interval in which to search for roots
    :param n_samples: int, optional. number of samples distributed uniformly in window
    :return: Level_Cross: a namedtuple with members
        level = float the iso level that is crossed and
        solutions = list of scipy.optimize.RootResults
    """

    if np.isscalar(iso_levels):
        return _find_iso_points_single_level(f, iso_levels, window=window, n_samples=n_samples)
    else:
        return [_find_iso_points_single_level(f, level, window=window, n_samples=n_samples) for level in iso_levels]


def find_roots(f, iso_levels, window, n_samples=100):
    """
    find the points where a scalar function crosses a specified horizontal line within a specified interval.
    :param f: callable with float or numpy.array input
    :param iso_levels: scalar or array_like: horizontal level for which to solve
    :param window: tuple (xmin,xmax) defining the interval in which to search for roots
    :param n_samples: int, optional. number of samples distributed uniformly in window
    :return: Level_Cross: a namedtuple with members
        level = float the iso level that is crossed and
        solutions = list of scalars

    Raises RootConvergenceError if a solution fails to converge
    """

    if np.isscalar(iso_levels):
        return _find_roots_single_level(f, iso_levels, window=window, n_samples=n_samples)
    else:
        return [_find_roots_single_level(f, level, window=window, n_samples=n_samples) for level in iso_levels]


def roots(f, window, n_samples=100, samples=None):
    """
        A simpler find_roots with better interface
        Find the roots of a scalar function

        Parameters
        ----------
        f: callable
            The function.

        window: (xmin, xmax),
            The domain in which to look for points.

        n_samples: int, optional
            The number of samples.
            Ignored of `samples` is not None.
            The samples will be distributed
            uniformly in the `window` domain. The algorithm will track
            down roots, when it detects a change of sing between adjacent
            samples. This means that roots that are close by may be missed
            if the `n_samples` is too small.

            Alternatively, a sequence of pre calculated samples can be
            passed through the `samples` variable.
            Default is 100.

        samples: (x_s, y_s), optional
            If not None, skip the sampling step and use the
            samples passed `here` instead.
            Default is None

    """
    if samples is not None:
        x = np.asarray(samples[0])
        y = np.asarray(samples[1])

    else:
        x, y = sample_fun(f, window, n_samples)

    brackets = adj.zero_cross_brackets(x, y)

    for x_bracket, _ in brackets:
        sol = optimize.root_scalar(f, bracket=x_bracket)
        yield sol.root, sol.converged

import numpy as np
from itertools import tee


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def different_sign(a, b):
    """
        Test if a and b have different signs, or if a is zero an b is
        nonzero. Element-wise

    Parameters:
    -----------
        a, b: scalar or array_like

    Returns:
    --------
        out: boolean, scalar or array like
             output is True if a and b have different signs, or if a is zero an
             b is nonzero. Element-wise
    """
    if not np.isscalar(a):
        a = np.asanyarray(a)
        b = np.asanyarray(b)

    return ((a <= 0) & (b > 0)) | ((a >= 0) & (b < 0))


def zero_cross_brackets(x, y, transform=None):
    """
        Iterator of a 2-tuple sequence
        (xb1, yb1), (xb2, yb2), ...
        where 'xb' and 'yb' are 2-tuples that contain
        a zero crossing of y.

        Motivation:
        -----------
        'x', 'y' are samples of the function
        'y = f(x)'.
        The tuples returned are to be interpreted as
            'xb' = (xbelow, xabove)
            'yb' = (ybelow, yabove)
        where the interval [xbelow, xabove) contains a root of 'f(x)'.

        Note that the interval is closed from below and open from above

        Iteration stops when the shortest input iterable is exhausted.


        Parmeters:
        ----------
            x, y: sequence
                The elements `y_e` of `y` must be comparable to zero.
                If this is not the case, then a `transform` must be
                provided, where `transform(y_e)` is comparable to zero.
                See documentation for `transform` for an example.

            transform: callable, optional
                If `transform` is not None, it is applied to the
                elements `y_e` of `y`, before comparing them to zero.
                For example, when `y` samples a vector function and
                we are interested in the intersections with the
                `y_e[0] == 0` plane, then `transform(y_e)` should return
                `y_e[0]`.
                Default is None


    """
    if transform is None:
        def _id(x):
            return x
        transform = _id

    def crosses(yb):
        tyb = (transform(x) for x in yb)
        return different_sign(*tyb)

    xb = _pairwise(x)
    yb = _pairwise(y)
    return filter(lambda s: crosses(s[1]), zip(xb, yb))

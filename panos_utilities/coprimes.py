from math import gcd
from fractions import Fraction

version = '0.1'


def pyramid(n_max):
    """
    iterable of all pairs of positive integers (x,y),
    :param n_max: positive integer
    """
    return ((x, y) for y in range(1, n_max + 1) for x in range(1, y))


def coprime_pyramid(n_max):
    """
    iterabe of all pairs of coprime positive integers up to n_max
    :param n_max: positive integer: maximum allowed integer in the domain
    :return: generator yielding tuples of positive integers (x,y) so that x<y and y<=n_max with gcd(x,y) == 1

     >>> list(coprime_pyramid(0))
    []
    >>> list(coprime_pyramid(3))
    [(1, 2), (1, 3), (2, 3)]
    >>> list(coprime_pyramid(4))
    [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4)]
    >>> list(coprime_pyramid(5))
    [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5)]
    """
    return ((x, y) for x, y in pyramid(n_max) if gcd(x, y) == 1)


def coprime_fractions(n_max):
    """
    iterable of all fractions built from integers up in the domain  [1, n_max]
    :param n_max: positive integer: maximum allowed integer in the domain
    :return: sorted list of all fractions built from integers up in the domain  [1, n_max]

    usage:
    >>> list(coprime_fractions(-2))
    []
    >>> list(coprime_fractions(4))
    [Fraction(1, 2), Fraction(2, 1), Fraction(1, 3), Fraction(3, 1), Fraction(2, 3), Fraction(3, 2), Fraction(1, 4),
    Fraction(4, 1), Fraction(3, 4), Fraction(4, 3), Fraction(1, 1)]
    >>> list(coprime_fractions(3))
    [Fraction(1, 2), Fraction(2, 1), Fraction(1, 3), Fraction(3, 1), Fraction(2, 3), Fraction(3, 2), Fraction(1, 1)]
    >>> print(*sorted(coprime_fractions(5)),sep=' ')
    1/5 1/4 1/3 2/5 1/2 3/5 2/3 3/4 4/5 1 5/4 4/3 3/2 5/3 2 5/2 3 4 5
    """

    for x, y in coprime_pyramid(n_max):
        yield Fraction(x, y)
        yield Fraction(y, x)

    if n_max > 0:
        yield Fraction(1)

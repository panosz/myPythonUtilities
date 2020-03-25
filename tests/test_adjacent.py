#  import numpy as np
import numpy.testing as nt
#  import pytest
from panos_utilities import adjacent as adj


def test_different_sign_scalar():
    assert adj.different_sign(-1, 1)
    assert adj.different_sign(0, 1)
    assert adj.different_sign(0, -1)

    assert not adj.different_sign(1, 1)
    assert not adj.different_sign(-1, -1)
    assert not adj.different_sign(0, 0)
    assert not adj.different_sign(1, 0)
    assert not adj.different_sign(-1, 0)


def test_different_sign_iterable_all():
    a = (-1, 0, 0, 1)
    b = (1, 1, -1, -1)
    assert all(adj.different_sign(a, b))


def test_zero_cross_brackets_single_bracket():
    x = range(2)
    y = [-1, 1]

    xb_expected = tuple(x)
    yb_expected = tuple(y)

    brackets = list(adj.zero_cross_brackets(x, y))
    assert len(brackets) == 1

    xb, yb = brackets[0]
    nt.assert_array_equal(xb, xb_expected)
    nt.assert_array_equal(yb, yb_expected)


def test_zero_cross_brackets_one_element():
    x = range(1)
    y = [1]

    brackets = list(adj.zero_cross_brackets(x, y))
    assert len(brackets) == 0


def test_zero_cross_multiple_brackets():
    x = range(10)
    y = [1, 0, -1, -1, 1, 0]

    b_expected = [
                  ((1, 2), (0, -1)),
                  ((3, 4), (-1, 1))
                 ]
    brackets = list(adj.zero_cross_brackets(x, y))

    for b, be in zip(brackets, b_expected):
        xb, yb = b
        xb_expected, yb_expected = be
        nt.assert_array_equal(xb, xb_expected)
        nt.assert_array_equal(yb, yb_expected)


def test_zero_cross_2D_multiple_brackets():
    x = range(10)
    y = [[1, 10],
         [0, 20],
         [-1, 18],
         [-1, 9],
         [1, 8],
         [0, 3]]

    def first_component(y_e):
        return y_e[0]

    b_expected = [
                  ((1, 2), ([0, 20], [-1, 18])),
                  ((3, 4), ([-1, 9], [1, 8]))
                 ]
    brackets = list(adj.zero_cross_brackets(x, y, transform=first_component))

    for b, be in zip(brackets, b_expected):
        xb, yb = b
        xb_expected, yb_expected = be
        nt.assert_array_equal(xb, xb_expected)
        nt.assert_array_equal(yb, yb_expected)


def test_zero_cross_2D_empty():
    x = []
    y = []

    def first_component(y_e):
        return y_e[0]

    brackets = list(adj.zero_cross_brackets(x, y, transform=first_component))

    assert len(brackets) == 0


def test_zero_cross_2D_one_element():
    x = [1]
    y = [[1, 2]]

    def first_component(y_e):
        return y_e[0]

    brackets = list(adj.zero_cross_brackets(x, y, transform=first_component))

    assert len(brackets) == 0

import numpy as np


def different_sign(a, b):
    """
      :param a: number of numpy array
      :param b: number of numpy array
      :return: Returns true if a an b have different signs, or if a is zero an b is nonzero. Element-wise
      calculation for numpy arrays.
    """
    return ((a <= 0) & (b > 0)) | ((a >= 0) & (b < 0))


def zero_cross_boolean_index(a):
    """
    :param a: numpy array
    :return: array of boolean indices, indicating the first of two elements adjacent elements with different sign.
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


def zero_cross_elems(x, y):
    """
    :param x: numpy array (sorted)
    :param y: numpy array must have the same length with x
    :return: the values of x for which y crosses zero. The values returned are the ones before or exactly at
    the zero crossing.If the last value of x is a root, it does not get returned.
    """
    assert (x.size == y.size), "x and y must have equal size!"
    indices = np.arange(x.size)[zero_cross_boolean_index(y)]

    return x[indices], x[indices+1], y[indices], y[indices+1]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 100)
    y = np.sin(2 * np.pi * x)

    x_before,x_after,y_before,y_after = zero_cross_elems(x,y)

    plt.plot(x,y,'k-',marker='+')
    plt.plot(x_before,y_before,'ro')
    plt.plot(x_after,y_after,'rx')
    plt.show()
#
# print(zero_cross_elems(np.array([0, 2, 3, -1, 1, 0])))
# print(zero_cross_elems(np.array([-1, 1, 2])))

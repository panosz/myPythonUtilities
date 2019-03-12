import adjacent_numpy as adn
import numpy as np
import scipy.optimize as optimize


def find_root(f, window, n_samples=100):
    x0, x1 = window
    x = np.linspace(x0, x1, n_samples)
    y = f(x)
    x_before, x_after, _, _ = adn.zero_cross_elems(x, y)

    return [optimize.root_scalar(f, bracket=bra_ket) for bra_ket in zip(x_before, x_after)]


def find_iso_points(f, iso_level, *args, **kwargs):
    def f_moved(*args_inner, **kwargs_inner):
        return f(*args_inner, **kwargs_inner) - iso_level

    return find_root(f_moved, *args, **kwargs)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    my_window = (0, 10)


    def fun_to_solve(x):
        return np.sin(2 * np.pi * x)


    level = 0.5

    result = find_iso_points(fun_to_solve, iso_level=level, window=my_window)

    x_iso = [sol.root for sol in result]
    y_iso = [level] * len(x_iso)

    x0, x1 = my_window

    x = np.linspace(x0, x1, 1000)
    y = fun_to_solve(x)

    plt.plot(x, y, 'b-')
    plt.plot(x_iso, y_iso, 'rx')
    plt.show()

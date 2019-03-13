import numpy as np
import sys
import os

my_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, my_path)

import panos_utilities.roots as roots

if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def fun_to_solve(x):
        return np.sin(2 * np.pi * x)


    def sample_fun(f, window, n_samples):
        x0, x1 = window
        x = np.linspace(x0, x1, n_samples)
        return (x, f(x))


    print('Using roots version {}'.format(roots.version))

    level = 0.5
    window = (0, 10)

    result = roots.find_iso_points(fun_to_solve, level, window)

    x_iso = [sol.root for sol in result.solutions]
    y_iso = [result.level] * len(x_iso)

    x, y = sample_fun(fun_to_solve, window, n_samples=1000)

    plt.plot(x, y, 'b-')
    plt.plot(x_iso, y_iso, 'rx')
    plt.show()

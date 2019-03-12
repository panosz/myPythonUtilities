import numpy as np
import sys
import os
my_path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
print(my_path)
sys.path.insert(1,my_path)

import panos_utilities.roots as roots


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    my_window = (0, 10)


    def fun_to_solve(x):
        return np.sin(2 * np.pi * x)


    print(roots.version)

    level = 0.5

    result = roots.find_iso_points(fun_to_solve, iso_level=level, window=my_window)

    x_iso = [sol.root for sol in result]
    y_iso = [level] * len(x_iso)

    x0, x1 = my_window

    x = np.linspace(x0, x1, 1000)
    y = fun_to_solve(x)

    plt.plot(x, y, 'b-')
    plt.plot(x_iso, y_iso, 'rx')
    plt.show()


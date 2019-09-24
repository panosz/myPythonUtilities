import numpy as np
import panos_utilities
import panos_utilities.roots as roots
import matplotlib.pyplot as plt
import pkg_resources


def fun_to_solve(x):
    return np.sin(2 * np.pi * x)


panos_utilities_version = pkg_resources.require("panos_utilities")[0].version
print('Using panos_utilities version {}'.format(panos_utilities_version))
print('Using roots version {}'.format(roots.__version__))

level = [0.5, -0.4, 0, 1.2]
window = (0, 10)

x, y = roots.sample_fun(fun_to_solve, window, n_samples=1000)
plt.plot(x, y, 'b-')

results = roots.find_roots(fun_to_solve, level, window)

for result in results:
    x_iso = result.solutions
    y_iso = [result.level] * len(x_iso)
    plt.plot(x_iso, y_iso, 'rx')

plt.show()

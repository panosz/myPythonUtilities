import numpy as np
import sys
import os
my_path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
print(my_path)
sys.path.insert(1,my_path)

import panos_utilities.adjacent_numpy as adj

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 100)
    y = np.sin(2 * np.pi * x)

    result = adj.zero_cross_elems(x, y)

    plt.plot(x, y, 'k-', marker='+')
    plt.plot(result.x_before, result.y_before, 'ro')
    plt.plot(result.x_after, result.y_after, 'rx')
    plt.show()
#
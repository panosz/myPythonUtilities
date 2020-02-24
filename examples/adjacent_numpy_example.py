import numpy as np
import panos_utilities.adjacent_numpy as adj
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(2 * np.pi * x)

result = adj.zero_cross_elems(x, y)

plt.plot(x, y, 'k-', marker='+')
plt.plot(result.x_before, result.y_before, 'ro')
plt.plot(result.x_after, result.y_after, 'rx')
plt.show()

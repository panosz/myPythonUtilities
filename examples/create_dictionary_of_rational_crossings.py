import numpy as np
import sys
import os
from fractions import Fraction

my_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(my_path)
sys.path.insert(1, my_path)
import panos_utilities.roots as roots


def func_to_solve(x):
    return np.sin(x)


ratio_set = {Fraction(x, y) for x in range(-4, 4) if x != 0 for y in range(1, 4)}

iso_crosses = {}
for ratio in ratio_set:
    iso_crosses[ratio] = roots.find_iso_points(func_to_solve,
                                               iso_level=float(ratio),
                                               window=(0, 10), n_samples=100).solutions

for level, solutions in iso_crosses.items():
    root_string = ','.join([str(sol.root) for sol in solutions])
    print('crossing {} at {}'.format(level, root_string))

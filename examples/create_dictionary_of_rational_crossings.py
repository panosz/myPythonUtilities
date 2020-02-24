import numpy as np
from itertools import chain
import panos_utilities.roots as roots
import panos_utilities.coprimes as coprimes


def func_to_solve(x):
    return np.sin(x)


ratio_max_int = 5

ratio_set = sorted(chain.from_iterable((x, -x)
                   for x in coprimes.coprime_fractions(ratio_max_int)))

iso_crosses = {}
for ratio in ratio_set:
    iso_crosses[ratio] = roots.find_iso_points(func_to_solve,
                                               iso_levels=float(ratio),
                                               window=(0, 10),
                                               n_samples=100).solutions

for level, solutions in iso_crosses.items():
    root_string = ','.join([str(sol.root) for sol in solutions])
    print('crossing {} at {}'.format(level, root_string))

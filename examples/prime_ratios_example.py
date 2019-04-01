import sys
import os
from itertools import chain

my_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("added path in sys: {}\n".format(my_path))
sys.path.insert(1, my_path)

import panos_utilities.coprimes as coprimes

n_max = 4
print("coprime fractions from integers up to {}:".format(n_max))

fractions = chain.from_iterable((x, -x) for x in coprimes.coprime_fractions(n_max))

print(*sorted(fractions), sep=' ')

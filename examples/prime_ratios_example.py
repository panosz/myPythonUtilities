from itertools import chain
import panos_utilities.coprimes as coprimes

n_max = 4
print("coprime fractions from integers up to {}:".format(n_max))

fractions = chain.from_iterable((x, -x) for x in
                                coprimes.coprime_fractions(n_max))

print(*sorted(fractions), sep=' ')

import numpy as np
import random
import itertools
from inspect import signature

def normalize(array):
    array = np.array(array)
    return array / sum(array)

def get_partition(n):
    """Gets the partition"""
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
            x = a[(k - 1)] + 1
            k -= 1
            while 2 * x <= y:
                    a[k] = x
                    y -= x
                    k += 1
            l = k + 1
            while x <= y:
                    a[k] = x
                    a[l] = y
                    yield a[:k + 2]
                    x += 1
                    y -= 1
            a[k] = x + y
            y = x + y - 1
            yield a[:k + 1]

def select_partition(n, type = 'equal', one_allowed='yes'):
    partition = list(get_partition(n))
    if one_allowed == 'no':
        partition = partition[:-1]
    if np.shape(partition) == (1,1): return partition[0]
    if type == 'equal':
        # print(np.shape(partition))
        return np.random.choice(partition)
    elif type == 'rounded':
        p = normalize([1 / len(i) for i in partition])
        return np.random.choice(partition, p = p)
    # choice = np.random.randint(0, len(p))
    # return p[choice]


def dc_alg(choices, epochs, alpha=1.0, weights=0, counts=0, verbosity=0):
    selections = []
    if np.all(counts == 0):
        counts = [
         1] * len(choices)
    weights = np.array(weights)
    if np.all(weights) == 0:
        weights = [
         1] * len(choices)
    for q in range(epochs):
        sum_ = sum([weights[i] * counts[i] ** alpha for i in range(len(choices))])
        probs = [weights[i] * counts[i] ** alpha / sum_ for i in range(len(choices))]
        selection_index = np.random.choice((list(range(len(choices)))), p=probs)
        counts = [i + 1 for i in counts]
        counts[selection_index] = 0
        selections.append(choices[selection_index])

    selections = np.array(selections)
    counts = np.array(counts)
    if verbosity == 0:
        return selections
    if verbosity == 1:
        return (
         selections, counts)

def normal_distribution_maker(bins):
    distribution = np.random.normal(size=100000)
    distribution = np.histogram(distribution, bins=bins, density=True)[0]
    distribution /= np.sum(distribution)
    return distribution

def nPVI(d):
    m = len(d)
    return 100 / (m - 1) * sum([abs((d[i] - d[(i + 1)]) / (d[i] + d[(i + 1)]) / 2) for i in range(m - 1)])

def nPVI_averager(window_width, durs):
    return [nPVI(durs[i:i + window_width]) for i in range(len(durs) - window_width)]

def nCVI(d):
    matrix = [list(i) for i in itertools.combinations(d, 2)]
    matrix = [nPVI(i) for i in matrix]
    return sum(matrix) / len(matrix)

def segment(num_of_segments,nCVI_average,factor=2.0):
    section_durs = factor ** np.random.normal(size=2)
    while abs(nCVI(section_durs) - nCVI_average) > 1.0:
        section_durs = factor ** np.random.normal(size=2)
    for i in range(num_of_segments - 2):
        next_section_durs = np.append(section_durs,[factor ** np.random.normal()])
        ct=0
        while abs(nCVI(next_section_durs) - nCVI_average) > 1.0:
            ct+=1
            next_section_durs = np.append(section_durs, [factor ** np.random.normal()])
        section_durs = next_section_durs
        # print(ct)
    section_durs /= np.sum(section_durs)
    return section_durs

def auto_args(target):
    """
    A decorator for automatically copying constructor arguments to `self`.
    """
    # Get a signature object for the target method:
    sig = signature(target)
    def replacement(self, *args, **kwargs):
        # Parse the provided arguments using the target's signature:
        bound_args = sig.bind(self, *args, **kwargs)
        # Save away the arguments on `self`:
        for k, v in bound_args.arguments.items():
            if k != 'self':
                setattr(self, k, v)
        # Call the actual constructor for anything else:
        target(self, *args, **kwargs)
    return replacement

def spread(init, max_ratio):
    exponent = np.clip(np.random.normal() / 3, -1, 1)
    return init * (max_ratio ** exponent)

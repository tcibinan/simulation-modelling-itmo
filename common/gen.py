import logging

import scipy
import scipy.stats
from math import sqrt


class LinearCongruentialGenerator:

    def __init__(self, ec, m=2 ** 31 - 1, a=630360016):
        self.m = m
        self.a = a
        self.ec = ec

    def __next__(self):
        self.ec = (self.a * self.ec) % self.m
        return self.ec / self.m

    def __call__(self):
        return self.__next__()


class IntegerGenerator:

    def __init__(self, probs, lcg):
        self.probs = probs
        self.lcg = lcg

    def __next__(self):
        generated_prob = self.lcg()
        collected_prob = 0
        for (integer, prob) in self.probs.items():
            collected_prob += prob
            if collected_prob > generated_prob:
                return integer
        logging.warning('Taking the last available integer because collected probability differs '
                        'with the generated one:\n%s\n%s'
                        % (collected_prob, generated_prob))
        return self.probs.items[:-1][1]

    def __call__(self):
        return self.__next__()

    @property
    def M(self):
        return sum([integer * prob for integer, prob in self.probs.items()])

    @property
    def D(self):
        quad_items = {integer ** 2: prob for integer, prob in self.probs.items()}
        quad_ig = IntegerGenerator(quad_items, self.lcg)
        return quad_ig.M - self.M ** 2


class UniformGenerator:

    def __init__(self, lcg, min, max):
        self._lcg = lcg
        self._min = min
        self._max = max

    def __next__(self):
        return self._min + self._lcg() * (self._max - self._min)

    def __call__(self):
        return self.__next__()


class NormalGenerator:

    def __init__(self, lcg, mu, sigma):
        self._lcg = lcg
        self._mu = mu
        self._sigma = sigma

    def __next__(self):
        return scipy.stats.norm.ppf(self._lcg()) * self._sigma + self._mu

    def __call__(self):
        return self.__next__()

import numpy as np


class FormFilter:

    def __init__(self, a, b, c, x_gen, v_gen):
        self.a = a
        self.b = b
        self.c = c
        self.x_gen = x_gen
        self.v_gen = v_gen
        self.x = np.array([])
        self.v = np.array([])
        self.y = np.array([])

    def model(self, max_step):
        np.append(self.x, self._generate_X())
        np.append(self.v, self._generate_V())
        while self.y.size() < max_step:
            np.append(self.y, self.c * self.x[-1])
            np.append(self.x, self.a * self.x[-1] + self.b * self.v[-1])
            np.append(self.v, self._generate_V())

    def _generate_X(self):
        return np.array([self.x_gen() for _ in range(0, self.a.shape[1])])

    def _generate_V(self):
        return np.array([self.v_gen() for _ in range(0, self.v.shape[0])])


class ExponentiallyCorrelatedFilter:

    def __init__(self, delta, t, x_gen, v_gen):
        self._delta = delta
        self._t = t
        self._x_gen = x_gen
        self._v_gen = v_gen
        self.y = np.array([])
        self.v = np.array([])

    def model(self, max_step):
        self.y = np.append(self.y, self._x_gen())
        self.v = np.append(self.v, self._v_gen())
        b0 = np.sqrt(1 - np.exp(-2 * self._delta / self._t))
        a1 = 1.0
        a0 = - np.exp(- self._delta / self._t)
        while self.y.size < max_step:
            new_y = -a0 / a1 * self.y[-1] + b0 / a1 * self.v[-1]
            self.y = np.append(self.y, new_y)
            new_v = self._v_gen()
            self.v = np.append(self.v, new_v)

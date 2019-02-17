class LinearCongruentialGenerator:

    def __init__(self, ec, m=2 ** 31 - 1, a=630360016):
        self.m = m
        self.a = a
        self.ec = ec

    def __next__(self):
        self.ec = (self.a * self.ec) % self.m
        return self.ec / self.m

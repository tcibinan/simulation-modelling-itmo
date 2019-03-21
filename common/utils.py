from math import sqrt


def confidence_interval(M, D, n, t):
    e = t * sqrt(D / n)
    return M - e, M + e

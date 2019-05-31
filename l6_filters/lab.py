import logging
from datetime import datetime
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

from common.gen import LinearCongruentialGenerator, UniformGenerator, NormalGenerator
from common.log import init_logging
from l6_filters.filter import ExponentiallyCorrelatedFilter


def main():
    init_logging(file='logs/l6-output-%s.log' % datetime.now(), debug=True)

    # Конфигурации
    initial_value = 9645730
    lcg = LinearCongruentialGenerator(initial_value)
    initial_gen = NormalGenerator(lcg, mu=1, sigma=2)
    max_step = 10000
    plot_steps = 100
    correlation_steps = 50
    experiments_number = 100
    show_white_noise_distributions = False
    show_corellated_noise_distributions = False
    hypothesis_a = 0.1
    hypothesis_k = lambda count: 1.73 * count ** (1 / 3)

    # Исходные данные
    sigma = np.sqrt(4)
    delta = 3.0
    t = 50.0

    # 1. Генераторы дискретного белого шума
    uniform_interval = np.sqrt(12 * (sigma ** 2)) / 2
    uniform_gen = UniformGenerator(lcg, min=-uniform_interval, max=uniform_interval)
    normal_gen = NormalGenerator(lcg, mu=0, sigma=sigma)

    if show_white_noise_distributions:
        uniform_noise = [uniform_gen() for _ in range(0, 10000)]
        normal_noise = [normal_gen() for _ in range(0, 10000)]

        uniform_noise_histogram = np.histogram(uniform_noise, bins=30, range=(-3, 3), density=True)
        normal_noise_histogram = np.histogram(normal_noise, bins=30, range=(-12, 12), density=True)

        _, (left_ax, right_ax) = plt.subplots(1, 2)
        left_ax.bar(range(plot_steps), uniform_noise[:plot_steps])
        right_ax.plot(uniform_noise_histogram[1][:-1], uniform_noise_histogram[0])
        right_ax.set_ylim(0, 0.4)
        plt.show()

        _, (left_ax, right_ax) = plt.subplots(1, 2)
        left_ax.bar(range(plot_steps), normal_noise[:plot_steps])
        right_ax.plot(normal_noise_histogram[1][:-1], normal_noise_histogram[0])
        right_ax.set_ylim(0, 0.4)
        plt.show()

    # 2. Формирующий фильтр
    if show_corellated_noise_distributions:
        uniform_filter = ExponentiallyCorrelatedFilter(delta=delta, t=t, x_gen=initial_gen,
                                                       v_gen=uniform_gen)
        uniform_filter.model(max_step=max_step)
        uniform_filter_histogram = np.histogram(uniform_filter.y, bins=30, range=(-5, 5),
                                                density=True)
        normal_filter = ExponentiallyCorrelatedFilter(delta=delta, t=t, x_gen=initial_gen,
                                                      v_gen=normal_gen)
        normal_filter.model(max_step=max_step)
        normal_filter_histogram = np.histogram(normal_filter.y, bins=30, range=(-5, 5),
                                               density=True)

        _, (left_ax, right_ax) = plt.subplots(1, 2)
        left_ax.bar(range(plot_steps), uniform_filter.y[:plot_steps])
        right_ax.plot(uniform_filter_histogram[1][:-1], uniform_filter_histogram[0])
        plt.show()

        _, (left_ax, right_ax) = plt.subplots(1, 2)
        left_ax.bar(range(plot_steps), normal_filter.y[:plot_steps])
        right_ax.plot(normal_filter_histogram[1][:-1], normal_filter_histogram[0])
        plt.show()

    uniform_filters = [ExponentiallyCorrelatedFilter(delta=delta, t=t, x_gen=initial_gen,
                                                     v_gen=uniform_gen)
                       for _ in range(experiments_number)]
    normal_filters = [ExponentiallyCorrelatedFilter(delta=delta, t=t, x_gen=initial_gen,
                                                    v_gen=normal_gen)
                      for _ in range(experiments_number)]

    def auto_correlation(x, length):
        return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, length)])

    logging.info('Performing %s experiments for uniform filter' % experiments_number)
    for index, filter in enumerate(uniform_filters):
        if index and index % 10 == 0:
            logging.info('Experiments has been performed %s/%s' % (index, experiments_number))
        filter.model(max_step=max_step)
    logging.info('All experiments has been performed for uniform filter')
    logging.info('Collecting autocorrelation, M and D for uniform filter')
    uniform_correlation = reduce(np.add, [auto_correlation(filter.y, correlation_steps)
                                          for filter in uniform_filters]) \
                          / experiments_number
    uniform_outputs = reduce(np.append, [filter.y for filter in uniform_filters])
    uniform_M = np.mean(uniform_outputs)
    uniform_M2 = np.mean(uniform_outputs ** 2)
    uniform_D = uniform_M2 - uniform_M ** 2
    logging.info('M = %s' % uniform_M)
    logging.info('D = %s' % uniform_D)

    logging.info('Performing %s experiments for normal filter' % experiments_number)
    for index, filter in enumerate(normal_filters):
        if index and index % 10 == 0:
            logging.info('Experiments has been performed %s/%s' % (index, experiments_number))
        filter.model(max_step=max_step)
    logging.info('All experiments has been performed for normal filter')
    logging.info('Collecting autocorrelation, M and D for normal filter')
    normal_correlation = reduce(np.add, [auto_correlation(filter.y, correlation_steps)
                                         for filter in normal_filters]) \
                         / experiments_number
    normal_outputs = reduce(np.append, [filter.y for filter in normal_filters])
    normal_M = np.mean(normal_outputs)
    normal_M2 = np.mean(normal_outputs ** 2)
    normal_D = normal_M2 - normal_M ** 2
    logging.info('M = %s' % normal_M)
    logging.info('D = %s' % normal_D)

    logging.info('Plotting autocorrelation functions')
    plt.plot(range(correlation_steps), uniform_correlation)
    plt.plot(range(correlation_steps), normal_correlation)
    plt.show()

    def spectrum(w, correlation):
        return 2 * np.sum([np.cos(w * k) * correlation[k] for k in range(len(correlation))])

    spectrum_dots = np.linspace(0, 1, correlation_steps)
    uniform_spectrum = [spectrum(w, uniform_correlation) for w in spectrum_dots]
    normal_spectrum = [spectrum(w, normal_correlation) for w in spectrum_dots]

    logging.info('Plotting spectrum functions')
    plt.plot(spectrum_dots, uniform_spectrum)
    plt.plot(spectrum_dots, normal_spectrum)
    plt.show()

    logging.info('Testing distributions hypothesis')

    def normal_f(x, mu, sigma):
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) \
               / np.sqrt(2 * np.pi * sigma ** 2)

    def interval_Z(interval_prob, interval_count, total_count):
        return (interval_count - total_count * interval_prob) ** 2 / (total_count * interval_prob)

    hypothesis_intervals = np.linspace(-6, 6, plot_steps)
    uniform_Z = 0
    normal_Z = 0
    for interval_start, interval_end in zip(hypothesis_intervals[:-1], hypothesis_intervals[1:]):
        interval_hypothesis_prob = integrate.quad(func=lambda x: normal_f(x, mu=0, sigma=sigma),
                                                  a=interval_start, b=interval_end)[0]
        interval_uniform_count = np.logical_and(uniform_outputs >= interval_start,
                                                uniform_outputs < interval_end).sum()
        interval_normal_count = np.logical_and(normal_outputs >= interval_start,
                                               normal_outputs < interval_end).sum()
        uniform_Z += interval_Z(interval_hypothesis_prob, interval_uniform_count,
                                uniform_outputs.size)
        normal_Z += interval_Z(interval_hypothesis_prob, interval_normal_count,
                               normal_outputs.size)

    from scipy.stats.distributions import chi2

    def X2(a, k):
        return chi2.ppf(q=1 - a, df=k - 1)

    uniform_X2 = X2(hypothesis_a, hypothesis_k(uniform_outputs.size))
    normal_X2 = X2(hypothesis_a, hypothesis_k(normal_outputs.size))

    logging.info('Uniform noise Z  = %s' % uniform_Z)
    logging.info('Uniform noise X2 = %s' % uniform_X2)
    logging.info('Normal noise  Z  = %s' % normal_Z)
    logging.info('Normal noise  X2 = %s' % normal_X2)


if __name__ == '__main__':
    main()

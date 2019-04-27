import logging
import operator
from datetime import datetime
from functools import reduce
from math import log

import matplotlib.pyplot as plt
import sympy
from sympy import Symbol
import numpy as np

from common.gen import LinearCongruentialGenerator
from common.log import init_logging
from l3_pure_birth.birth import PureBirth

init_logging(file='logs/l3-output-%s.log' % datetime.now())


class TimeGenerator:

    def __init__(self, lambda_, lcg):
        self.lambda_ = lambda_
        self.lcg = lcg

    def __call__(self, k):
        return - log(self.lcg()) / self.lambda_(k)


def main():
    # Последовательности интенсивности поступления заявок
    k = Symbol('k', real=True)
    lambdas = [
        sympy.Float(0.95, 2),
        0.15 * k ** 2 + 4,
        6.25 / (k + 5)
    ]

    # Конфигурации
    initial_value = 6450435
    lcg = LinearCongruentialGenerator(initial_value)
    modelling_states = 11
    max_modelling_states = 50
    experiments_number = 200
    plot_steps = 200
    plot_transitions = [
        np.linspace(0, 15, plot_steps),
        np.linspace(0, 1, plot_steps),
        np.linspace(0, 30, plot_steps)
    ]

    # Диаграммы состояний процессов
    logging.info('Generating initial state diagrams for all models...')
    fig, axes = plt.subplots(1, 3)
    for index, lambda_ in enumerate(lambdas):
        pb = PureBirth(tg=TimeGenerator(lambda_=lambda arg: lambda_.subs(k, arg), lcg=lcg))
        pb.model(max_steps=modelling_states)
        ax1 = axes[index]
        ax1.plot(pb.transitions, pb.states, drawstyle='steps')
        ax1.set_title('lambda=%s' % lambda_)
    plt.show()

    # Прогоны каждой из моделей
    for lambda_, all_transitions in zip(lambdas, plot_transitions):
        logging.info('Starting to process a model with lambda=%s.' % lambda_)
        pbs = [PureBirth(tg=TimeGenerator(lambda_=lambda arg: lambda_.subs(k, arg), lcg=lcg))
               for _ in range(experiments_number)]
        logging.info('Performing %s experiments...' % experiments_number)
        for pb in pbs:
            pb.model(max_transition=all_transitions[len(all_transitions)-1],
                     max_steps=max_modelling_states)
        all_states = []
        logging.info('Collecting transitions for %s periods...' % len(all_transitions))
        for transition in all_transitions:
            states = []
            for pb in pbs:
                state = None
                for index, pb_transition in enumerate(pb.transitions):
                    if pb_transition > transition:
                        state = index - 1
                        break
                states.append(state if state is not None else len(pb.transitions))
            all_states.append(states)

        logging.info('Calculating state probabilities...')
        # Расчет вероятностей каждого из событий
        all_probs = []
        for state in range(modelling_states):
            all_probs.append([states.count(state) / len(states) if states else 0
                              for states in all_states])
        prob_plots = reduce(operator.add, [[all_transitions, probs] for probs in all_probs])

        logging.info('Calculating M and D...')
        # Расчет математического ожидания и дисперсии
        all_M = []
        all_D = []
        for states in all_states:
            M = np.sum(states) / len(states) if states else 0
            M2 = np.sum(np.array(states) ** 2) / len(states) if states else 0
            all_M.append(M)
            all_D.append(M2 - M ** 2)

        logging.info('Generating plot for state probabilities, M and D...')
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(*prob_plots)
        ax1.legend(['P(X(t)=%s)' % state for state in range(modelling_states)], loc='upper center',
                   ncol=2)
        ax2.plot(all_transitions, all_M, all_transitions, all_D)
        ax2.legend(['M(X)', 'D(X)'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    main()

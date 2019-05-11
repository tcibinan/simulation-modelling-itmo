import logging
import operator
from datetime import datetime
from functools import reduce

import matplotlib.pyplot as plt
import sympy
from sympy import Symbol
import numpy as np

from common.gen import LinearCongruentialGenerator
from common.log import init_logging
from l3_pure_birth.lab import TimeGenerator
from l4_birth_death.birth_death import BirthAndDeath


def main():
    init_logging(file='logs/l4-output-%s.log' % datetime.now())

    # Последовательности интенсивности поступления заявок
    k = Symbol('k', real=True)
    lambdas = [
        sympy.Float(0.4, 1),
        1.15 * k ** 3 + 1,
        0.7 / (k + 12) ** 2
    ]
    mus = [
        sympy.Float(0.3, 1),
        0.33 * k ** 3,
        5 / k ** 2
    ]
    intensities = [_lambda + mu for (_lambda, mu) in zip(lambdas, mus)]

    # Конфигурации
    initial_value = 6450435
    numbers_after_dot = 5
    lcg = LinearCongruentialGenerator(initial_value)
    modelling_states = 11
    max_modelling_states = 1000
    pi_epsilon = 0.00001
    experiments_number = 200
    plot_steps = 200
    plot_transitions = [
        np.linspace(0, 15, plot_steps),
        np.linspace(0, 0.35, plot_steps),
        np.linspace(0, 100, plot_steps)
    ]

    # Диаграммы состояний процессов
    logging.info('Generating initial state diagrams for all models...')
    fig, axes = plt.subplots(1, 3)
    for index, funcs in enumerate(zip(lambdas, mus, intensities)):
        lambda_, mu, intensity = funcs
        pb = BirthAndDeath(lcg=lcg,
                           tg=TimeGenerator(intensity=lambda arg: intensity.subs(k, arg),
                                            lcg=lcg),
                           lambda_=lambda arg: lambda_.subs(k, arg),
                           mu=lambda arg: mu.subs(k, arg))
        pb.model(max_iterations=modelling_states)
        ax1 = axes[index]
        ax1.plot(pb.transitions, pb.states, drawstyle='steps')
        ax1.set_title('lambda=%s\nmu=%s' % (lambda_, mu))
    plt.show()

    # Прогоны каждой из моделей
    for lambda_, mu, intensity, all_transitions \
            in zip(lambdas, mus, intensities, plot_transitions):
        logging.info('Starting to process a model with lambda=%s and mu=%s.' % (lambda_, mu))
        pbs = [BirthAndDeath(lcg=lcg,
                             tg=TimeGenerator(intensity=lambda arg: intensity.subs(k, arg),
                                              lcg=lcg),
                             lambda_=lambda arg: lambda_.subs(k, arg),
                             mu=lambda arg: mu.subs(k, arg))
               for _ in range(experiments_number)]
        logging.info('Performing %s experiments...' % experiments_number)
        for index, pb in enumerate(pbs):
            if index % 10 == 0:
                logging.debug('%s experiments were performed...' % index)
            pb.model(max_transition=all_transitions[len(all_transitions) - 1],
                     max_steps=max_modelling_states)
        logging.debug('All experiments were performed.')
        all_states = []
        logging.info('Collecting transitions for %s periods...' % len(all_transitions))
        for index, transition in enumerate(all_transitions):
            if index % 10 == 0:
                logging.debug('%s transitions were collected...' % index)
            states = []
            for pb in pbs:
                state = None
                for pb_index, pb_transition in enumerate(pb.transitions):
                    if pb_transition > transition:
                        state = pb.states[pb_index]
                        break
                states.append(state if state is not None else pb.states[-1])
            all_states.append(states)
        logging.debug('All transitions were collected.')

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
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.plot(*prob_plots)
        ax1.legend(['P(X(t)=%s)' % state for state in range(modelling_states)], loc='upper center',
                   ncol=2)
        ax2.plot(all_transitions, all_M)
        ax2.legend(['M(X)'], loc='upper left')
        ax3.plot( all_transitions, all_D)
        ax3.legend(['D(X)'], loc='upper left')
        plt.show()

        # Расчет финальных вероятностей
        pi = [1]
        for state in range(1, max_modelling_states):
            numerator = 1
            denominator = 1
            for inner_state in range(1, state):
                numerator *= lambda_.subs(k, inner_state)
                denominator *= mu.subs(k, inner_state)
            pi.append(numerator / denominator)
            if state > modelling_states and abs(pi[-1] - pi[-2]) < pi_epsilon:
                break
        pi_sum = sum(pi)
        logging.info('Listing final probabilities calculated for first %s states...' % len(pi))
        for state in range(modelling_states):
            state_p = pi[state]/pi_sum
            logging.info('p%s =\t%s' % (state, np.around(float(state_p), numbers_after_dot)))


if __name__ == '__main__':
    main()

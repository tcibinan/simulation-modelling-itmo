import logging
import operator
from datetime import datetime
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.functions.elementary.exponential import log
from scipy.special import gamma

from common.gen import LinearCongruentialGenerator
from common.log import init_logging
from l5_recovery.recovery import Recovery


class WeibullGenerator:

    def __init__(self, lcg, alpha, lambda_):
        self._lcg = lcg
        self._F = Symbol('t', real=True)
        self._gen = 1 / lambda_ * (-log(self._F)) ** (1 / alpha)

    def __next__(self):
        return self._gen.subs(self._F, self._lcg())

    def __call__(self):
        return self.__next__()


def main():
    init_logging(file='logs/l5-output-%s.log' % datetime.now(), debug=True)

    # Последовательности параметров альфа и лямбда
    params = [
        {
            'gen': {
                'alpha': 1.0,
                'lambda_': 7.0
            },
            'max_transition': 0.5
        },
        {
            'gen': {
                'alpha': 0.1,
                'lambda_': 4.0
            },
            'max_transition': 2.0
        },
        {
            'gen': {
                'alpha': 2.0,
                'lambda_': 10.0
            },
            'max_transition': 0.25
        }
    ]
    # Конфигурации
    initial_value = 6450435
    numbers_after_dot = 5
    lcg = LinearCongruentialGenerator(initial_value)
    max_step = 10
    experiments_number = 1000
    plot_steps = 30

    # Предварительные прогоны
    logging.info('Generating initial state diagrams for all models...')
    fig, axes = plt.subplots(1, 3)
    for index, model_params in enumerate(params):
        model = Recovery(gen=WeibullGenerator(lcg, **model_params['gen']))
        model.model(max_steps=max_step)
        state_pairs = zip(model.states[:-1], model.states[:-1])
        transition_pairs = zip(model.transitions[:-1], model.transitions[1:])
        ax = axes[index]
        for transition_states, transitions_H in zip(state_pairs, transition_pairs):
            ax.plot(transitions_H, transition_states, drawstyle='steps', color='blue')
            ax.set_title('alpha=%s\nlambda=%s'
                         % (model_params['gen']['alpha'], model_params['gen']['lambda_']))
        ax.grid()
    plt.show()

    # Прогоны каждой из моделей
    logging.info('Performing %s experiments for all models.' % experiments_number)
    for model_params in params:
        lambda_ = model_params['gen']['lambda_']
        alpha = model_params['gen']['alpha']
        logging.info('Performing experiments for the model with alpha=%s and lambda=%s.'
                     % (alpha, lambda_))
        models = [Recovery(gen=WeibullGenerator(lcg, **model_params['gen']))
                  for _ in range(0, experiments_number)]
        for index, model in enumerate(models):
            if index and index % 10 == 0:
                logging.debug('Experiments performed: %s/%s.' % (index, experiments_number))
            model.model(max_transition=model_params['max_transition'])
        logging.debug('All experiments has finished.')

        logging.info('Collecting transitions for %s intervals.' % plot_steps)
        transitions_H = np.linspace(0, model_params['max_transition'], plot_steps)
        states = []
        for index, transition in enumerate(transitions_H):
            if index and index % 10 == 0:
                logging.debug('Transitions collected: %s/%s.' % (index, plot_steps))
            transition_states = []
            for model in models:
                state = None
                for transition_index, model_transition in enumerate(model.transitions):
                    if model_transition > transition:
                        state = model.states[transition_index]
                        break
                transition_states.append(state if state is not None else model.states[-1])
            states.append(transition_states)
        logging.debug('All transitions have been collected.')

        logging.info('Calculating M and D for ksi.')
        all_transitions = reduce(operator.add, [model.generated_transitions for model in models])
        M = (np.sum(all_transitions) / len(all_transitions)).__float__()
        M2 = np.sum(np.array(all_transitions) ** 2) / len(all_transitions)
        D = (M2 - M ** 2).__float__()
        M_theoretical = 1 / lambda_ * gamma(1 + 1 / alpha)
        D_theoretical = (1 / lambda_) ** 2 * gamma(1 + 2 / alpha) - M_theoretical ** 2
        logging.info('experimental M = %s.' % (np.around(M, numbers_after_dot)))
        logging.info('theoretical  M = %s.' % (np.around(M_theoretical, numbers_after_dot)))
        logging.info('experimental D = %s.' % (np.around(D, numbers_after_dot)))
        logging.info('theoretical  D = %s.' % (np.around(D_theoretical, numbers_after_dot)))

        logging.info('Calculating H(t).')
        # Расчет функции восстановления
        H = []
        for transition_states in states:
            transition_M = np.sum(transition_states) / len(transition_states) \
                if transition_states else 0
            H.append(transition_M)
        logging.info('Plotting H(t).')
        plt.plot(transitions_H, H)
        plt.title('H(t)')
        plt.show()

        logging.info('Calculating experimental F(t).')
        F = []
        transitions_F = np.linspace(0, model_params['max_transition'], plot_steps)
        generated_transitions = [model.generated_transitions for model in models]
        generated_transitions = np.array(list(reduce(np.append, generated_transitions)))
        for transition in transitions_F:
            transition_F = (generated_transitions <= transition).sum() / generated_transitions.size
            F.append(transition_F)
        logging.info('Calculating theoretical F(t).')
        F_theoretical = 1 - np.exp(-(lambda_ * transitions_F) ** alpha)

        logging.info('Calculating f(t).')
        f = []
        for index, transition in enumerate(transitions_F[:-1]):
            left_condition = generated_transitions > transition
            next_transition = transitions_F[index + 1]
            right_condition = generated_transitions <= next_transition
            number_of_transitions = np.logical_and(left_condition, right_condition).sum()
            transition_f = number_of_transitions / generated_transitions.size / (
                        next_transition - transition)
            f.append(transition_f)
        f.append(0)
        f = np.array(f)
        logging.info('Calculating theoretical f(t).')
        f_theoretical = alpha * lambda_ * (lambda_ * transitions_F) ** (alpha - 1) \
                        * np.exp(-(lambda_ * transitions_F) ** alpha)

        logging.info('Calculating G(t).')
        G = 1 - np.array(F)
        logging.info('Calculating theoretical G(t).')
        G_theoretical = np.exp(-(lambda_ * transitions_F) ** alpha)
        G[G == 0] = G[G != 0].min()

        logging.info('Calculating phi(t).')
        phi = f / G
        logging.info('Calculating theoretical phi(t).')
        phi_theoretical = alpha * lambda_ * (lambda_ * transitions_F) ** (alpha - 1)

        logging.info('Plotting F(t), f(t), G(t) and phi(t).')
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].plot(transitions_F, F)
        axes[0, 0].plot(transitions_F, F_theoretical)
        axes[0, 0].set_title('F(t)')
        axes[0, 1].plot(transitions_F + (transitions_F[1] - transitions_F[0]), f)
        axes[0, 1].plot(transitions_F, f_theoretical)
        axes[0, 1].set_title('f(t)')
        axes[1, 0].plot(transitions_F, G)
        axes[1, 0].plot(transitions_F, G_theoretical)
        axes[1, 0].set_title('G(t)')
        axes[1, 1].plot(transitions_F, phi)
        axes[1, 1].plot(transitions_F, phi_theoretical)
        axes[1, 1].set_title('phi(t)')
        plt.show()


if __name__ == '__main__':
    main()

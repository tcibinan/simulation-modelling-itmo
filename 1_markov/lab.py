import logging
from operator import concat
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np

from common.gen import LinearCongruentialGenerator
from common.log import init_logging
from lab1.markov import MarkovChain

init_logging()


def show_transition_diagrams(models, corner_states):
    fig, axes = plt.subplots(2, 2)
    for index, model in enumerate(models):
        ax = axes[int(index / 2), index % 2]
        ax.plot(model.transitions, drawstyle='steps')
        ax.set_ylim(corner_states)
        ax.set_title('Initial state: %s' % model.initial_state)
    plt.show()


def main():
    # Матрица переходных состояний
    G = [
        [0.3, 0.3, 0, 0.4],
        [0.7, 0.3, 0, 0],
        [0, 0, 0.4, 0.6],
        [0, 0, 0.2, 0.8]
    ]

    # Конфигурации
    states = range(0, 4)
    modelling_steps = 200
    models_number = 100
    initial_value = 42352531
    numbers_after_dot = 5

    # Генераторы случайных величин
    lcg = LinearCongruentialGenerator(initial_value)

    # 2. Прогоны модели с начальными значениями, соответствующими состояниям дискретной модели.
    initial_models = [MarkovChain(G, lcg, state) for state in states]

    for index, model in enumerate(initial_models):
        logging.info('%s/%s models has finished.' % (index + 1, len(initial_models)))
        model.model(modelling_steps)

    # 3. Отображение диаграмм состояния дискретных марковских цепей.
    show_transition_diagrams(initial_models, [states[0] - 1, states[-1] + 1])

    # 5. Эксперименты для определения стационарного распределения вероятностей дискретной
    # марковской цепи.
    models = [MarkovChain(G, lcg, state) for state in states]
    models = models * (int(models_number / len(states)) + 1)
    models = models[:models_number]
    for index, model in enumerate(models):
        logging.info('%s/%s models has finished.' % (index + 1, len(models)))
        model.model(modelling_steps)

    all_transitions = reduce(concat, [model.transitions for model in models])

    stationary_probabilities = [all_transitions.count(state) / len(all_transitions)
                                for state in states]
    logging.info('Experiments stationary probabilities = %s.'
                 % list(np.around(stationary_probabilities, numbers_after_dot)))

    # 6. Расчет стационарного распределения вероятностей дискретной марковской цепи.
    mG = []
    for state in states:
        mg = [g[state] for g in G]
        mg[state] -= 1
        mG.append(mg)
    mG[-1] = [1, 1, 1, 1]
    a = np.array(mG)
    b = np.array([0, 0, 0, 1])
    x = np.linalg.solve(a, b)
    logging.info('Calculated stationary probabilities = %s.'
                 % list(np.around(x, numbers_after_dot)))


if __name__ == '__main__':
    main()

import logging
from datetime import datetime

from sympy import Symbol, solve
import numpy as np

from common.gen import LinearCongruentialGenerator, IntegerGenerator
from common.log import init_logging
from common.utils import confidence_interval
from l2_branching_process.branching import BranchingProcess

init_logging(file='logs/l1-output-%s.log' % datetime.now())


def main():
    # Производящая функция
    s = Symbol('s', real=True)
    p0 = Symbol('p0')
    p2 = Symbol('p2')
    p3 = Symbol('p3')
    p4 = Symbol('p4')
    p5 = Symbol('p5')
    p6 = Symbol('p6')
    phi = p0 + p2 * (s ** 2) + p3 * (s ** 3) + p4 * (s ** 4) + p5 * (s ** 5) + p6 * (s ** 6)

    # Конфигурации
    initial_value = 42352531
    numbers_after_dot = 5
    modelling_steps = 10
    experiments_number = 100
    K = range(0, 9)
    t = 1.964

    # Генераторы случайных величин
    lcg = LinearCongruentialGenerator(initial_value)

    # 1. Выбор распределений вероятности
    m = phi.diff(s).subs(s, 1)
    logging.info('m = %s' % m)

    pre_critical_distribution_law = {p0: 0.85, p2: 0.05, p3: 0.04, p4: 0.03, p5: 0.02, p6: 0.01}
    critical_distribution_law = {p0: 0.60, p2: 0.3, p3: 0.04, p4: 0.03, p5: 0.02, p6: 0.01}
    post_critical_distribution_law = {p0: 0.65, p2: 0.11, p3: 0.09, p4: 0.07, p5: 0.05, p6: 0.03}

    distribution_laws = [
        ('Pre-critical distribution', pre_critical_distribution_law),
        ('Critical distribution', critical_distribution_law),
        ('Post-critical distribution', post_critical_distribution_law)
    ]

    for (name, law) in distribution_laws:
        logging.info(name)
        logging.info('Distribution law %s' % law)
        logging.info('sum(p) = %s' % np.around(sum(law.values()), numbers_after_dot))
        logging.info('m = %s' % np.around(float(m.subs(law)), numbers_after_dot))

        # 2. Определение математического ожидания, дисперсии и вероятности вырождения ветвящегося
        #    процесса
        ig = IntegerGenerator(dict(zip([0, 2, 3, 4, 5, 6], law.values())), lcg)
        M = ig.M
        D = ig.D
        MX1 = M
        DX1 = D ** 2
        real_roots = solve(phi.subs(law) - s)
        logging.debug('Q = phi(Q), roots: %s' % real_roots)
        Q = sorted(real_roots)[0]
        logging.info('M = %s' % np.around(M, numbers_after_dot))
        logging.info('D = %s' % np.around(D, numbers_after_dot))
        logging.info('MX1 = %s' % np.around(MX1, numbers_after_dot))
        logging.info('DX1 = %s' % np.around(DX1, numbers_after_dot))
        logging.info('Q = %s' % np.around(float(Q), numbers_after_dot))

        # 3. Моделирование ветвящегося процесса
        processes = [BranchingProcess(ig) for _ in range(experiments_number)]
        all_probabilities = []
        for index, process in enumerate(processes):
            process.model(num_steps=modelling_steps)
            states = process.states
            logging.debug('Branching process %s/%s has finished after %s steps.' % (index + 1,
                                                                                    len(processes),
                                                                                    len(states)))
            probabilities = [states.count(k) / len(states) for k in K]
            logging.debug('Branching process states %s%s.' % ('...' if len(states) > 5 else '',
                                                              ', '.join(map(str, states[-5:]))))
            logging.debug('Branching process probabilities %s.' % probabilities)
            all_probabilities.append(probabilities)
        probabilities_map = [[] for _ in K]

        for k in K:
            for probabilities in all_probabilities:
                probabilities_map[k].append(probabilities[k])

        for k in K:
            probabilities = probabilities_map[k]
            Mp = sum(probabilities) / len(probabilities)
            Mp2 = sum([prob ** 2 for prob in probabilities]) / len(probabilities)
            Dp = Mp2 - Mp ** 2
            interval = confidence_interval(Mp, Dp, len(probabilities), t)
            # TODO 23.03.19: Математическое ожидание для K=1 слишком высокое, при том, что p1 = 0
            logging.info('K = %s; '
                         'M = %s; '
                         'D = %s; '
                         'Confidence interval = %s'
                         % (k,
                            np.around(Mp, numbers_after_dot),
                            np.around(Dp, numbers_after_dot),
                            np.around(interval, numbers_after_dot)))


if __name__ == '__main__':
    main()

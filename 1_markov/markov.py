import logging


class MarkovChain:

    def __init__(self, G, lcg, initial_state):
        self.G = G
        self.lcg = lcg
        self.initial_state = initial_state
        self.state = self.initial_state
        self.transitions = [self.state]

    def model(self, num_steps):
        for step in range(num_steps):
            transitions = self.G[self.state]
            available_transitions = [(prob, state) for (state, prob) in enumerate(transitions)
                                     if prob > 0]
            self.state = self._get_next_state(available_transitions)
            self.transitions.append(self.state)

    def _get_next_state(self, available_transitions):
        generated_prob = self.lcg.__next__()
        collected_prob = 0
        for (prob, state) in available_transitions:
            collected_prob += prob
            if collected_prob > generated_prob:
                return state
        logging.warning('Taking the last available state because collected probability differs '
                        'with generated one:\n%s\n%s'
                        % (collected_prob, generated_prob))
        return available_transitions[:-1][1]

class BirthAndDeath:

    def __init__(self, lcg, tg, lambda_, mu):
        self.lcg = lcg
        self.tg = tg
        self.states = [0]
        self.transitions = [0]
        self.lambda_ = lambda_
        self.mu = mu

    def model(self, max_steps=None, max_transition=None):
        while True:
            latest_transition = self.transitions[-1]
            if max_transition and latest_transition >= max_transition \
                    or max_steps and len(self.transitions) >= max_steps:
                return
            transition = self.tg(len(self.transitions))
            self.transitions += [self.transitions[-1] + transition]
            current_state = self.states[-1]
            if not current_state:
                self.states += [current_state + 1]
            else:
                lambda_ = self.lambda_(current_state)
                mu = self.mu(current_state)
                generated_value = self.lcg() * (lambda_ + mu)
                if generated_value < lambda_:
                    self.states += [current_state + 1]
                elif current_state > 0:
                    self.states += [current_state - 1]
                else:
                    self.states += [current_state]

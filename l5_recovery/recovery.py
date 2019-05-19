class Recovery:

    def __init__(self, gen):
        self.gen = gen
        self.states = [0]
        self.transitions = [0]
        self.generated_transitions = []

    def model(self, max_steps=None, max_transition=None):
        while True:
            latest_transition = self.transitions[-1]
            if max_transition and latest_transition >= max_transition \
                    or max_steps and self.states[-1] >= max_steps:
                return
            generated_value = self.gen()
            self.generated_transitions += [generated_value]
            self.transitions += [self.transitions[-1] + generated_value]
            self.states += [self.states[-1] + 1]

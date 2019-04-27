class PureBirth:

    def __init__(self, tg):
        self.tg = tg
        self.states = [0]
        self.transitions = [0]

    def model(self, max_steps=None, max_transition=None):
        while True:
            latest_transition = self.transitions[len(self.transitions) - 1]
            if max_transition and latest_transition >= max_transition \
                    or max_steps and len(self.transitions) >= max_steps:
                return
            transition = self.tg(len(self.transitions))
            self.transitions += [self.transitions[len(self.transitions) - 1] + transition]
            self.states += [len(self.states)]

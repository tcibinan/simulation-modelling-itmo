class PureBirth:

    def __init__(self, tg):
        self.tg = tg
        self.states = [0]
        self.transitions = [0]

    def model(self, num_states):
        while True:
            if len(self.transitions) >= num_states:
                return
            transition = self.tg(len(self.transitions))
            self.transitions += [self.transitions[len(self.transitions) - 1] + transition]
            self.states += [len(self.states)]

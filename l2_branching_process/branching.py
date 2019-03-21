import logging


class BranchingProcess:

    def __init__(self, ig):
        self.ig = ig
        self.state = 1
        self.states = [self.state]

    def model(self, num_steps):
        for step in range(num_steps - 1):
            logging.debug(self.state)
            self.state = sum([self.ig() for _ in range(0, self.state)])
            self.states.append(self.state)
            if self.state == 0:
                return

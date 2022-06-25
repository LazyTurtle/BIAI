from mesa.time import BaseScheduler
from mesa import Model


class DelegatedScheduler(BaseScheduler):
    # This class delegate the use of the step function to another class or function, providing the agent calling it
    def __init__(self, model: Model, delegate: callable) -> None:
        super(DelegatedScheduler, self).__init__(model)
        assert callable(delegate), f"The delegate method might not be callable: {delegate}"
        self.delegate = delegate

    def step(self):
        for agent in self.agent_buffer(shuffled=False):
            self.delegate(agent=agent)
        self.steps += 1
        self.time += 1


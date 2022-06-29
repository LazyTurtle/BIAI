from mesa.time import BaseScheduler
from mesa import Model, Agent


class TagScheduler(BaseScheduler):
    def __init__(self, model: Model) -> None:
        super(TagScheduler, self).__init__(model)
        self.tagged_agents = dict()

    def add(self, agent: Agent, tag=None):
        super(TagScheduler, self).add(agent)
        if tag is not None:
            if tag in self.tagged_agents.keys():
                self.tagged_agents[tag].append(agent)
            else:
                self.tagged_agents[tag] = [agent]

    def step(self, tag=None):
        if tag is None:
            super(TagScheduler, self).step()
        else:
            for agent in self.tagged_agents[tag]:
                agent.step()
            self.steps += 1
            self.time += 1

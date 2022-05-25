from mesa import Model, Agent
from mesa.space import SingleGrid
import numpy as np


class ResourceModel(Model):

    def __init__(self, width, height, num_collectors, num_resources, num_gathering_points=1):
        self.grid = SingleGrid(width, height, True)
        self.num_collectors = num_collectors
        self.num_resources = num_resources
        self.num_gathering_points = num_gathering_points
        self.running = True
        self.agents = list()
        self.setup()

    def setup(self):
        self.fill_env(Resource, self.num_resources)
        self.fill_env(Collector, self.num_collectors)
        self.fill_env(GatheringPoint, self.num_gathering_points)

    def fill_env(self, agent_cl, n):
        for i in range(n):
            agent = agent_cl(i, self)
            self.agents.append(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            while(not self.grid.is_cell_empty((x,y))):
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def step(self) -> None:
        for agent in self.agents:
            agent.step()


class Collector(Agent):
    def __init__(self, unique_id, model, proximity_distance=2):
        super(Collector, self).__init__(unique_id, model)
        self.proximity_distance = proximity_distance
        self.prox_shape = (self.proximity_distance * 2 + 1, self.proximity_distance * 2 + 1)
        self.proximity = np.zeros(self.prox_shape)

    def update_proximity_information(self):
        # reset, 1 means that you can move freely there
        self.proximity.fill(1)
        neighbours = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.proximity_distance)

        center = int((self.proximity_distance * 2 + 1) / 2)
        self.proximity[center, center] = 0

        for agent in neighbours:
            x = (agent.pos[0] - self.pos[0]) + center
            y = (agent.pos[1] - self.pos[1]) + center
            # the grid might be toroidal, so we have to wrap up if agents are on the edges
            x = x % self.prox_shape[0]
            y = y % self.prox_shape[1]

            if type(agent) is Collector:
                self.proximity[y, x] = 0
            if type(agent) is Resource:
                self.proximity[y, x] = 0.5
            if type(agent) is GatheringPoint:
                self.proximity[y, x] = 0.5

        # the browser grid cells are indexed by [x][y]
        # where [0][0] is assumed to be the bottom-left and [width-1][height-1] is the top-right
        # so it's the opposite representation of numpy
        self.proximity = np.flip(self.proximity, 0)

    def step(self) -> None:
        self.update_proximity_information()


    def portrayal(self):
        shape = {
            "Shape": "circle",
            "Color": "red",
            "Filled": "true",
            "Layer": 0,
            "r": 0.5}
        return shape


class Resource(Agent):
    def __init__(self, unique_id, model):
        super(Resource, self).__init__(unique_id, model)

    def portrayal(self):
        shape = {
            "Shape": "circle",
            "Color": "green",
            "Filled": "true",
            "Layer": 0,
            "r": 0.5}
        return shape


class GatheringPoint(Agent):
    def __init__(self, unique_id, model):
        super(GatheringPoint, self).__init__(unique_id, model)

    def portrayal(self):
        shape = {
            "Shape": "rect",
            "Color": "black",
            "Filled": "true",
            "Layer": 0,
            "w": 1,
            "h": 1
        }
        return shape

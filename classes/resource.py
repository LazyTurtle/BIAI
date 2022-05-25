from mesa import Model, Agent
from mesa.space import MultiGrid


class ResourceModel(Model):

    def __init__(self, width, height, num_collectors, num_resources, num_gathering_points=1):
        self.grid = MultiGrid(width, height, True)
        self.num_collectors = num_collectors
        self.num_resources = num_resources
        self.num_gathering_points = num_gathering_points
        self.running = True
        self.setup()

    def setup(self):
        self.fill_env(Resource, self.num_resources)
        self.fill_env(Collector, self.num_collectors)
        self.fill_env(GatheringPoint, self.num_gathering_points)

    def fill_env(self, agent_cl, n):
        for i in range(n):
            agent = agent_cl(i, self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))


class Collector(Agent):
    def __init__(self, unique_id, model):
        super(Collector, self).__init__(unique_id, model)

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

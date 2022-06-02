from mesa import Agent
import numpy as np
import source.resource
import math


class Collector(Agent):

    def __init__(self, unique_id, model, proximity_distance=1, vision_distance=3):
        super(Collector, self).__init__(unique_id, model)
        self.proximity_distance = proximity_distance
        self.prox_shape = (self.proximity_distance * 2 + 1, self.proximity_distance * 2 + 1)
        self.proximity = np.zeros(self.prox_shape)
        self.vision_distance = vision_distance
        self.vision = np.zeros((3, 3))
        self.resource_sensor = np.zeros((3, 3))

        # data used during the evolution, to setup at each ResourceModel instantiation
        self.neural_network = None
        self.genome = None
        self.resources = 0
        self.points = 0

    def evolution_setup(self, neural_network):
        self.neural_network = neural_network
        self.resources = 0
        self.points = 0

    def get_action(self, update_sensors=True):
        if update_sensors:
            self.update_sensors()
        input_data = self.get_sensor_data()
        # The inputs are flattened in order to both have a list (required by neat) and to follow the order defined
        # by the coordinates of hyper neat
        input_data = input_data.flatten()
        output = self.neural_network.activate(input_data)
        action = np.argmax(output)
        return action

    def update_sensors(self):
        self.update_proximity_information()
        self.update_vision_information()
        self.update_resource_sensor()

    def update_proximity_information(self):
        # reset, 1 means that you can move freely there
        self.proximity.fill(1)
        neighbours = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.proximity_distance)

        self.proximity[self.proximity_distance, self.proximity_distance] = 0
        for agent in neighbours:
            j, i = self.array_indexes(agent, self.proximity_distance)

            if type(agent) is Collector:
                self.proximity[i, j] = 0
            if type(agent) is source.resource.Resource:
                self.proximity[i, j] = 0.5
            if type(agent) is source.resource.GatheringPoint:
                self.proximity[i, j] = 0.5

        # the browser grid cells are indexed by [x][y]
        # where [0][0] is assumed to be the bottom-left and [width-1][height-1] is the top-right
        # it's the opposite vertical representation of numpy
        self.proximity = np.flip(self.proximity, 0)

    def update_vision_information(self):

        self.vision.fill(0)
        neighbours = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.vision_distance)
        for agent in neighbours:
            if type(agent) is not source.resource.Resource:
                continue

            x, y, distance = self.model.relative_distances(self, agent)
            max_env_distance = max(self.model.grid.width, self.model.grid.height)
            food_distance = 1 - (distance / max_env_distance)
            bearings = math.atan2(y, x)
            mx = math.cos(bearings)
            my = math.sin(bearings)
            rx = round(mx)
            ry = round(my)
            i = ry + int(self.vision.shape[0] / 2)
            j = rx + int(self.vision.shape[0] / 2)
            self.vision[i, j] += food_distance
        self.vision = np.flip(self.vision, 0)
        self.vision = self.vision.clip(0, 1)

    def update_resource_sensor(self):
        self.resource_sensor.fill(0)
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=1)
        for agent in neighbors:
            if hasattr(agent, 'resources'):
                n_resources = agent.resources
                j, i = self.array_indexes(agent, 1)
                self.resource_sensor[i, j] = n_resources
        self.resource_sensor = np.flip(self.resource_sensor, 0)

    @staticmethod
    def topology():
        proximity_inputs = \
            [(-1., 1., -1.), (0., 1., -1.), (1., 1., -1.),
             (-1., 0., -1.), (0., 0., -1.), (1., 0., -1.),
             (-1., -1., -1.), (0., -1., -1.), (1., -1., -1.)]
        vision_inputs = \
            [(-1., 1., -0.5), (0., 1., -0.5), (1., 1., -0.5),
             (-1., 0., -0.5), (0., 0., -0.5), (1., 0., -0.5),
             (-1., -1., -0.5), (0., -1., -0.5), (1., -1., -0.5)]
        resources_inputs = \
            [(-1., 1., 0.), (0., 1., 0.), (1., 1., 0.),
             (-1., 0., 0.), (0., 0., 0.), (1., 0., 0.),
             (-1., -1., 0.), (0., -1., 0.), (1., -1., 0.)]
        inputs = proximity_inputs + vision_inputs + resources_inputs
        hidden_layers = \
            [
                [(-1., 1., 0.25), (0., 1., 0.25), (1., 1., 0.25),
                 (-1., 0., 0.25), (0., 0., 0.25), (1., 0., 0.25),
                 (-1., -1., 0.25), (0., -1., 0.25), (1., -1., 0.25),
                 (-1., 1., 0.5), (0., 1., 0.5), (1., 1., 0.5),
                 (-1., 0., 0.5), (0., 0., 0.5), (1., 0., 0.5),
                 (-1., -1., 0.5), (0., -1., 0.5), (1., -1., 0.5)]
            ]
        outputs = \
            [(-1., 1., 1.), (0., 1., 1.), (1., 1., 1.),
             (-1., 0., 1.), (0., 0., 1.), (1., 0., 1.),
             (-1., -1., 1.), (0., -1., 1.), (1., -1., 1.)]
        return inputs, hidden_layers, outputs

    def fitness(self):
        return self.points

    def array_indexes(self, neighbour, radius):
        dx, dy, _ = self.model.relative_distances(self, neighbour)
        x = dx + radius
        y = dy + radius
        return x, y

    def step(self) -> None:
        self.update_sensors()

    def portrayal(self):
        shape = {
            "text": f"id:{self.unique_id}",
            "text_color": "black",
            "Shape": "circle",
            "Color": "red",
            "Filled": "true",
            "Layer": 0,
            "r": 0.5}
        return shape

    def get_sensor_data(self):
        return np.array((self.proximity, self.vision, self.resource_sensor))

    def log(self):
        print(f"id:{self.unique_id}")
        print(self.proximity)
        print(self.vision)
        print(self.resource_sensor)

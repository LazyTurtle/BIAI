import random
import src.resource
import math
import logging
import numpy as np

from mesa import Agent


class Collector(Agent):

    def __init__(self, unique_id, model, proximity_distance=1, vision_distance=4, debug=False):
        super(Collector, self).__init__(unique_id, model)
        self.proximity_distance = proximity_distance
        self.prox_shape = (self.proximity_distance * 2 + 1, self.proximity_distance * 2 + 1)
        self.proximity = np.zeros(self.prox_shape)
        self.resources_vision_distance = vision_distance
        self.gathering_points_vision_distance = 10
        self.resources_vision = None
        self.gathering_vision = None
        self.debug = debug

        # data used during the evolution, to set up at each ResourceModel instantiation
        self.neural_network = None
        self.activations = None
        self.genome = None
        self.resources = 0
        self.points = 0

    def step(self):
        self.update_sensors()
        action = self.get_action()
        if self.debug:
            logging.info(f"action chosen: {action}")
        self.model.calculate_action_outcome(self, action)

    def evolution_setup(self, neural_network, genome):
        self.neural_network = neural_network
        self.genome = genome
        i, h, o = self.topology()
        self.activations = len(h) + 2
        self.resources = 0
        self.points = 0

    def get_action(self):
        input_data = self.get_sensor_data()
        if self.debug:
            logging.info(f"Collector {self.unique_id}")
            logging.info("From sensors:")
            logging.info(f"\n{input_data}")
        # The inputs are flattened in order to both have a list (required by neat) and to follow the order defined
        # by the coordinates of hyper neat
        input_data = input_data.flatten()
        output = None
        # This isn't strictly needed, but since it's a recurrent neural network and pureples examples do it
        # we are doing it as well, though it doesn't seem to have any effect on the evolution
        for _ in range(self.activations):
            output = self.neural_network.activate(input_data)
        if self.debug:
            logging.info(f"output activations: {output}")
        prob_mass = sum(output)
        action = None
        if prob_mass == 0.:
            action = random.choice(range(9))
        else:
            prob = np.array(output) / prob_mass
            action = np.random.choice(range(9), p=prob)

        return action

    def update_sensors(self):
        self.update_proximity()
        self.update_resource_vision()
        self.update_gathering_vision()

    def update_proximity(self):
        # reset, 1 means that you can move freely there
        self.proximity.fill(1)
        neighbours = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.proximity_distance)

        self.proximity[self.proximity_distance, self.proximity_distance] = 0 if self.resources == 0 else 1
        for agent in neighbours:
            j, i = self.array_indexes(agent, self.proximity_distance)

            if type(agent) is Collector:
                self.proximity[i, j] = 0
            if type(agent) is src.resource.GatheringPoint:
                self.proximity[i, j] = 0.25
            if type(agent) is src.resource.Resource:
                self.proximity[i, j] = 0.50

        # the browser grid cells are indexed by [x][y]
        # where [0][0] is assumed to be the bottom-left and [width-1][height-1] is the top-right
        # it's the opposite vertical representation of numpy
        self.proximity = np.flip(self.proximity, 0)

    def update_resource_vision(self):
        def resource_check(agent): return type(agent) == src.resource.Resource

        self.resources_vision = self.selective_vision(self.resources_vision_distance, resource_check)

    def update_gathering_vision(self):
        def gathering_check(agent): return type(agent) == src.resource.GatheringPoint

        self.gathering_vision = self.selective_vision(self.gathering_points_vision_distance, gathering_check)

    # is_visible should be a function that returns True if the argument is something that we want to see
    def selective_vision(self, vision_range, is_visible):
        vision = np.zeros((3, 3))

        neighbours = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=vision_range)
        for agent in neighbours:
            if not is_visible(agent):
                continue

            x, y, distance = self.model.relative_distances(self, agent)
            max_env_distance = max(self.model.grid.width, self.model.grid.height)
            agent_distance = 1 - (distance / max_env_distance)
            bearings = math.atan2(y, x)
            mx = math.cos(bearings)
            my = math.sin(bearings)
            rx = round(mx)
            ry = round(my)
            i = ry + int(vision.shape[0] / 2)
            j = rx + int(vision.shape[0] / 2)
            vision[i, j] += agent_distance

        vision = np.flip(vision, 0)  # np and mesa y-axis are opposite to each other
        vision = vision.clip(0, 1)
        return vision

    @staticmethod
    def topology():
        min_x = -1
        max_x = 1

        # for the inputs we have a 3x3 matrix for each sensor, and each sensor is a channel
        # we need a 3x9 matrix
        min_y = -1
        max_y = -0.63
        inputs = list()
        for y in np.linspace(max_y, min_y, 3):      # top-down
            for x in np.linspace(min_x, max_x, 9):  # left-right
                inputs.append((x, y))

        # the first hidden layer will be a 3x6 matrix
        min_y = -0.45
        max_y = -0.09
        hidden_layer_1 = list()
        for y in np.linspace(max_y, min_y, 3):
            for x in np.linspace(min_x, max_x, 6):
                hidden_layer_1.append((x, y))

        # the second hidden layer will be a 3x3 metrix
        min_y = 0.09
        max_y = 0.45
        hidden_layer_2 = list()
        for y in np.linspace(max_y, min_y, 3):
            for x in np.linspace(min_x, max_x, 3):
                hidden_layer_2.append((x, y))

        hidden_layers = [hidden_layer_2, hidden_layer_1]  # it's the same order of the example

        min_y = 0.63
        max_y = 1.0
        output = list()
        for y in np.linspace(max_y, min_y, 3):
            for x in np.linspace(min_x, max_x, 3):
                output.append((x, y))

        return inputs, hidden_layers, output

    def fitness(self):
        return self.points

    def array_indexes(self, neighbour, radius):
        dx, dy, _ = self.model.relative_distances(self, neighbour)
        x = dx + radius
        y = dy + radius
        return x, y

    def portrayal(self):
        # The documentation is wrong once again
        # Filled doesn't use strings "true"/"false" but the python True or False boolean
        shape = {
            # "text": f"id:{self.unique_id}",
            "text_color": "black",
            "Shape": "circle",
            "Color": "red",
            "Filled": False if self.resources == 0 else True,
            "Layer": 0,
            "r": 0.5}
        return shape

    def get_sensor_data(self):
        return np.array((self.proximity, self.resources_vision, self.gathering_vision))

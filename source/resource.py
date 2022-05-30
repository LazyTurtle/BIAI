import math
import numpy as np
from mesa import Model, Agent
from mesa.space import SingleGrid
from mesa.time import BaseScheduler


class ResourceModel(Model):

    def __init__(self, width, height, num_collectors, num_resources, num_gathering_points=1):
        super().__init__()
        self.grid = None
        self.width = width
        self.height = height
        self.num_collectors = num_collectors
        self.num_resources = num_resources
        self.num_gathering_points = num_gathering_points
        self.running = True
        self.schedule = None
        self.n_agents = 0
        self.reset()

    def reset(self):
        self.grid = SingleGrid(self.width, self.height, True)
        self.n_agents = 0
        self.schedule = BaseScheduler(self)
        self.fill_env(Resource, self.num_resources)
        self.fill_env(Collector, self.num_collectors)
        self.fill_env(GatheringPoint, self.num_gathering_points)

    def fill_env(self, agent_cl, n):
        for i in range(n):
            agent = agent_cl(self.n_agents, self)
            self.n_agents += 1
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            while not self.grid.is_cell_empty((x, y)):
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def step(self, **kwargs):
        if len(kwargs) == 0:
            self.schedule.step()
            return

        if "agent" in kwargs.keys() and "action" in kwargs.keys():
            agent = kwargs["agent"]
            action = kwargs["action"]
            y = int(action / 3)
            x = int(action % 3)
            dy = -(y - 1)  # mesa grid (bottom up) is vertically inverted in respect to numpy (top down)
            dx = x - 1
            new_x = agent.pos[0] + dx
            new_y = agent.pos[1] + dy
            new_pos = self.grid.torus_adj((new_x, new_y))
            if self.grid.is_cell_empty(new_pos):
                self.grid.move_agent(agent, new_pos)
                return 0, self.convergence()
            else:
                current_agent = self.grid[x, y]

                if type(current_agent) is Resource:
                    if agent.resources == 0:
                        agent.resources += 1
                        self.grid.remove_agent(current_agent)
                        self.grid.move_agent(agent, new_pos)
                        return 1, self.convergence()
                    else:
                        return 0, self.convergence()

                # TODO introduce a way to gather more than one resource at a time
                if type(current_agent) is GatheringPoint:
                    if agent.resources > 0:
                        agent.resources = 0
                        return 1, self.convergence()
                    else:
                        return 0, self.convergence()

                if type(current_agent) is Collector:
                    # for now, it's not planned to do anything in this case
                    return 0, self.convergence()

        return 0

    # takes into account the toroidal space
    def relative_distances(self, agent_a, agent_b):
        xdiff = agent_b.pos[0] - agent_a.pos[0]
        if xdiff > (self.grid.width / 2):
            xdiff = - self.grid.width + xdiff
        if xdiff < -(self.grid.width / 2):
            xdiff = self.grid.width + xdiff

        ydiff = agent_b.pos[1] - agent_a.pos[1]
        if ydiff > (self.grid.height / 2):
            ydiff = - self.grid.height + ydiff
        if ydiff < -(self.grid.height / 2):
            ydiff = self.grid.height + ydiff

        distance = math.sqrt(xdiff ** 2 + ydiff ** 2)

        return xdiff, ydiff, distance

    def agents(self, agent_cl=None):
        if agent_cl is None:
            return self.schedule.agents
        else:
            return [agent for agent in self.schedule.agents if type(agent) is agent_cl]

    # TODO define more sophisticated convergence criteria
    # define if we have converged
    def convergence(self):
        # for now let's just check if we gathered all resources
        return not any([type(a) == Resource for a in self.schedule.agents])


class Collector(Agent):

    def __init__(self, unique_id, model, proximity_distance=1, vision_distance=3):
        super(Collector, self).__init__(unique_id, model)
        self.proximity_distance = proximity_distance
        self.prox_shape = (self.proximity_distance * 2 + 1, self.proximity_distance * 2 + 1)
        self.proximity = np.zeros(self.prox_shape)
        self.vision_distance = vision_distance
        self.vision = np.zeros((3, 3))
        self.resource_sensor = np.zeros((3, 3))

        # data used during each iteration
        self.resources = 0
        self.points = 0

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
            if type(agent) is Resource:
                self.proximity[i, j] = 0.5
            if type(agent) is GatheringPoint:
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
            if type(agent) is not Resource:
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


class Resource(Agent):
    def __init__(self, unique_id, model):
        super(Resource, self).__init__(unique_id, model)
        self.resources = 1

    def portrayal(self):
        shape = {
            "text": f"id:{self.unique_id}",
            "text_color": "black",
            "Shape": "circle",
            "Color": "green",
            "Filled": "true",
            "Layer": 0,
            "r": 0.5}
        return shape

    def step(self) -> None:
        pass


class GatheringPoint(Agent):
    def __init__(self, unique_id, model):
        super(GatheringPoint, self).__init__(unique_id, model)

    def portrayal(self):
        shape = {
            "text": f"id:{self.unique_id}",
            "text_color": "white",
            "Shape": "rect",
            "Color": "black",
            "Filled": "true",
            "Layer": 0,
            "w": 1,
            "h": 1
        }
        return shape

    def step(self) -> None:
        pass

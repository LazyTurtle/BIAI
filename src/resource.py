import math
from mesa import Model, Agent
from mesa.space import SingleGrid
from .collector import Collector
from .tag_scheduler import TagScheduler


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
        self.schedule = TagScheduler(self)
        self.fill_env(Collector, self.num_collectors)
        self.fill_env(GatheringPoint, self.num_gathering_points)
        self.fill_env(Resource, self.num_resources)

    def fill_env(self, agent_cl, n):
        for i in range(n):
            agent = agent_cl(self.n_agents, self)
            self.n_agents += 1
            self.schedule.add(agent, agent_cl)
            self.grid.place_agent(agent, self.grid.find_empty())

    def step(self) -> bool:
        # for the moment we only need for collectors to make steps, not all agents
        self.schedule.step(Collector)
        has_converged = self.convergence()
        return has_converged

    def calculate_action_outcome(self, agent, action):
        # get the new absolute position
        y = int(action / 3)
        x = int(action % 3)
        dy = -(y - 1)  # mesa grid (bottom up) is vertically inverted in respect to numpy (top down)
        dx = x - 1
        new_x = agent.pos[0] + dx
        new_y = agent.pos[1] + dy
        new_pos = self.grid.torus_adj((new_x, new_y))

        if self.grid.is_cell_empty(new_pos):
            self.grid.move_agent(agent, new_pos)
            agent.points -= 1
        else:
            current_agent = self.grid[new_pos[0], new_pos[1]]

            if type(current_agent) is Resource:
                if agent.resources == 0:
                    agent.resources += 1
                    self.grid.remove_agent(current_agent)
                    self.grid.move_agent(agent, new_pos)

                    agent.points += 10

            # TODO introduce a way to gather more than one resource at a time
            if type(current_agent) is GatheringPoint:
                if agent.resources > 0:
                    agent.resources = 0

                    agent.points += 15
                else:
                    agent.points -= 10

            if type(current_agent) is Collector:
                # for now, it's not planned to do anything in this case
                pass

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


class GatheringPoint(Agent):
    def __init__(self, unique_id, model):
        super(GatheringPoint, self).__init__(unique_id, model)
        self.resources = -1

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

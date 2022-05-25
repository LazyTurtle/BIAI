from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer


def agent_portrayal(agent):
    return agent.portrayal()


def instantiate_server(grid, model):
    environment = grid
    pixels = 500
    environment_canvas = CanvasGrid(agent_portrayal, environment.width, environment.height, pixels, pixels)
    server = ModularServer(model,
                           [environment_canvas],
                           "Resource Model",
                           {"environment": grid, "num_collectors": 0, "num_resources": 20})
    server.port = 8521  # The default
    server.launch()

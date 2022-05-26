from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer


def agent_portrayal(agent):
    return agent.portrayal()


def instantiate_server(model):
    width = 10
    height = 10
    pixels = 500
    environment_canvas = CanvasGrid(agent_portrayal, width, height, pixels, pixels)
    server = ModularServer(model,
                           [environment_canvas],
                           "Resource Model",
                           {"width": width, "height": height, "num_collectors": 1, "num_resources": 5})
    server.port = 8521  # The default
    server.launch()

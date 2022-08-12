import logging
import mesa.visualization.modules
import evolution
import os
from datetime import datetime
from src.resource import ResourceModel
from src.collector import Collector
from mesa.visualization.ModularVisualization import ModularServer

#  NEAT_CONFIG_FILE_PATH = "config/NEAT.config"

def setup_logging():
    # filename = datetime.now().strftime('logs/log_%H_%M_%d_%m_%Y.log')
    filename = datetime.now().strftime('logs/log_%d_%m_%Y.log')
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        filemode="w+")  # I'm only interested in the last log of the day, so I just overwrite over the same file


def get_portrayal(a):
    return a.portrayal()


if __name__ == '__main__':
    setup_logging()

    cwd = os.getcwd()
    neat_config_file = os.path.join(cwd, NEAT_CONFIG_FILE_PATH)
    neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                     neat.DefaultStagnation, neat_config_file)

    best = evolution.best_agent()

    input_coo, hidden_coo, output_coo = Collector.topology()
    substrate = Substrate(input_coo, output_coo, hidden_coo)
    cppn = neat.nn.FeedForwardNetwork.create(best, neat_config)
    nn = create_phenotype_network(cppn, substrate)

    agent = Collector(0, None, debug=True)
    agent.evolution_setup(nn, best)
    canvas_width = 600
    canvas_height = 600

    grid = mesa.visualization.modules.CanvasGrid(
        get_portrayal, evolution.WIDTH, evolution.HEIGHT, canvas_width, canvas_height)

    server = ModularServer(
        ResourceModel, [grid], "Resource Gathering", {
            "num_collectors": [agent],
            "num_resources": evolution.NUM_RESOURCES,
            "num_gathering_points": evolution.NUM_GATHERING_POINTS,
            "width": evolution.WIDTH,
            "height": evolution.HEIGHT}
    )
    server.port = 8521  # The default
    server.launch()

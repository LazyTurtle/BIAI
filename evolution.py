import math
import logging
from datetime import datetime
logging.basicConfig(filename=datetime.now().strftime('logs/log_%H_%M_%d_%m_%Y.log'), level=logging.INFO)
import numpy as np
from src.resource import ResourceModel
from src.resource import Collector
import neat
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.substrate import Substrate


def evolve(genomes, config):
    # TODO glue together the code from pureples and mesa. the input is obtained by the agents and should be the list
    #  defined in input_coordinates, in that order

    # TODO create a configuration file to speedup testing
    steps = 500
    width = 100
    height = 100
    num_collectors = len(genomes)  # it might change from its predefined 20
    num_resources = 100*10
    num_gathering_points = 50

    trials_for_agent = 1

    assert num_collectors == len(
        genomes), f"The number of collectors ({num_collectors}) does not match the number of genomes ({len(genomes)})"

    environment = ResourceModel(width, height, 1, num_resources, num_gathering_points)
    input_coo, hidden_coo, output_coo = Collector.topology()
    substrate = Substrate(input_coo, output_coo, hidden_coo)

    for genome_id, genome in genomes:
        logging.info(f"Genome id: {genome_id}")
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        nn = create_phenotype_network(cppn, substrate)
        fitness = 0

        for t in range(trials_for_agent):
            logging.info(f"Trials {t}")
            environment.reset()
            nn.reset()
            collector = environment.agents(Collector)[0]
            collector.evolution_setup(nn)
            reward = 0

            for i in range(steps):
                points, has_converged = environment.step()
                reward += points
                if has_converged:
                    break

            logging.info(f"Reward for trial {t}: {reward}")
            logging.info(f"Converged in {i+1} steps")

            fitness += reward / math.sqrt((i+1)/steps)

        genome.fitness = fitness
        logging.info(f"fitness: {genome.fitness}")


if __name__ == '__main__':
    neat_config_file = "src/config/NEAT.config"
    generations = 200
    neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                     neat.DefaultStagnation, neat_config_file)
    pop = neat.population.Population(neat_config)
    best = pop.run(evolve, generations)

import math

import numpy as np
from source.resource import ResourceModel
from source.resource import Collector
import neat
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.substrate import Substrate
import random


def evolve(genomes, config):
    # TODO glue together the code from pureples and mesa. the input is obtained by the agents and should be the list
    #  defined in input_coordinates, in that order

    print("\n Evolving a new generation:")
    # divide genomes into n groups; for each group we will initialize an enviroment and test them inside it
    # the groups are heterogeneous (e.g: each member in the group has a different genome)
    group_size = 20
    start = 0
    end = group_size
    groups = list()
    random.shuffle(genomes)

    for group_index in range(len(genomes)//group_size):
        if group_index < len(genomes)//group_size:
            groups.append(genomes[start:end])
        else:
            groups.append(genomes[start:])
        start += group_size
        end += group_size

    for genome_group in groups:
        # TODO create a configuration file to speedup testing
        steps = 500
        width = 100
        height = 100
        num_collectors = len(genome_group)  # it might change from its predefined 20
        num_resources = 100*10
        num_gathering_points = 50
        assert num_collectors == len(
            genome_group), f"The number of collectors ({num_collectors}) does not match the number of genome_group ({len(genome_group)})"

        environment = ResourceModel(width, height, num_collectors, num_resources, num_gathering_points)
        input_coo, hidden_coo, output_coo = Collector.topology()
        substrate = Substrate(input_coo, output_coo, hidden_coo)

        collectors = environment.agents(Collector)
        for i in range(len(genome_group)):
            genome_id, genome = genome_group[i]
            cppn = neat.nn.FeedForwardNetwork.create(genome, config)
            nn = create_phenotype_network(cppn, substrate)
            agent = collectors[i]
            # nn.reset()  # in case we do multiple trials we have to reset the rnn before each of one
            agent.evolution_setup(nn)

        rewards = np.zeros(num_collectors)
        for i in range(steps):
            points, has_converged = environment.step()
            rewards += points
            if has_converged:
                break
        fitness = rewards  # / math.sqrt((i+1)/steps)
        print(f"Max: {'{:.2f}'.format(fitness.max())}, Mean: {'{:.2f}'.format(fitness.mean())}, Min: {'{:.2f}'.format(fitness.min())}")

        for i in range(len(environment.agents(Collector))):
            genome_id, genome = genome_group[i]
            genome.fitness = fitness[i]


if __name__ == '__main__':
    neat_config_file = "source/config/NEAT.config"
    generations = 200
    neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                     neat.DefaultStagnation, neat_config_file)
    pop = neat.population.Population(neat_config)
    best = pop.run(evolve, generations)

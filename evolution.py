import math

import numpy as np
from source.resource import ResourceModel
from source.resource import Collector
import neat
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.substrate import Substrate


def evolve(genomes, config):
    # TODO glue together the code from pureples and mesa. the input is obtained by the agents and should be the list
    #  defined in input_coordinates, in that order

    # TODO create a configuration file to speedup testing

    print ("Starting new generation \n")

    groups_amount = 5
    group_size = len(genomes) // groups_amount

    genome_groups = []
    group_start = 0
    group_end = group_size
    for i in range(groups_amount):
        if i == group_size - 1:
            genome_groups.append((genomes[group_start:]))
        else:
            genome_groups.append(genomes[group_start:group_end])
        group_start += group_size
        group_end += group_size

    for group in genome_groups:
        steps = 500
        width = 40
        height = 40
        num_collectors = len(group)  # it might change from its predefined 20
        num_resources = 100*10
        num_gathering_points = 500
        assert num_collectors == len(
            group), f"The number of collectors ({num_collectors}) does not match the number of genomes ({len(group)})"

        environment = ResourceModel(width, height, num_collectors, num_resources, num_gathering_points)
        input_coo, hidden_coo, output_coo = Collector.topology()
        substrate = Substrate(input_coo, output_coo, hidden_coo)

        collectors = environment.agents(Collector)

        for i in range(len(group)):
            genome_id, genome = group[i]
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
                print("Reached convergence before running out of steps!")
                break
        fitness = rewards  # / math.sqrt((i+1)/steps)
        print(fitness.max())

        for i in range(len(environment.agents(Collector))):
            genome_id, genome = group[i]
            genome.fitness = fitness[i]


if __name__ == '__main__':
    neat_config_file = "source/config/NEAT.config"
    generations = 20
    neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                     neat.DefaultStagnation, neat_config_file)
    pop = neat.population.Population(neat_config)
    best = pop.run(evolve, generations)

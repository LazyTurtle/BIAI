import math
import logging
from datetime import datetime
from src.resource import ResourceModel
from src.resource import Collector
import neat
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.substrate import Substrate

fit_max = list()
fit_mean = list()
def evolve(genomes, config):

    # TODO glue together the code from pureples and mesa. the input is obtained by the agents and should be the list
    #  defined in input_coordinates, in that order

    # TODO create a configuration file to speedup testing
    steps = 500
    width = 20
    height = 20
    num_collectors = len(genomes)  # it might change from its predefined 20
    num_resources = 20 * 4
    num_gathering_points = 10

    trials_for_agent = 1

    assert num_collectors == len(
        genomes), f"The number of collectors ({num_collectors}) does not match the number of genomes ({len(genomes)})"

    environment = ResourceModel(width, height, 1, num_resources, num_gathering_points)
    input_coo, hidden_coo, output_coo = Collector.topology()
    substrate = Substrate(input_coo, output_coo, hidden_coo)

    for genome_id, genome in genomes:
        # logging.info(f"Genome id: {genome_id}")
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        nn = create_phenotype_network(cppn, substrate)
        fitness = 0

        for t in range(trials_for_agent):
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

            # logging.info(f"Reward for trial {t}: {reward}")
            # logging.info(f"Converged in {i + 1} steps")

            fitness += reward / math.sqrt((i + 1) / steps)

        genome.fitness = fitness[0]
        # logging.info(f"Genome id {genome_id}, fitness: {genome.fitness}")
    fitnesses = [g.fitness for _, g in genomes]
    max_fitness = max(fitnesses)
    mean_fitness = sum(fitnesses)/len(fitnesses)
    fit_max.append(max_fitness)
    fit_mean.append(mean_fitness)
    logging.info(f"Generation {len(fit_max)}")
    logging.info(f"Mean fitness: {mean_fitness}")
    logging.info(f"Max fitness: {max_fitness}")


def setup_logging():
    # filename = datetime.now().strftime('logs/log_%H_%M_%d_%m_%Y.log')
    filename = datetime.now().strftime('logs/log_%d_%m_%Y.log')
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        filemode="w+")  # I'm only interested in the last log of the day, so I just overwrite over the same file


if __name__ == '__main__':
    setup_logging()
    neat_config_file = "src/config/NEAT.config"
    generations = 50
    neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                     neat.DefaultStagnation, neat_config_file)
    pop = neat.population.Population(neat_config)
    best = pop.run(evolve, generations)

    import matplotlib.pyplot as plt
    plt.plot(fit_max)
    plt.plot(fit_mean)
    plt.legend(["Max fitness", "Mean fitness"])
    plt.show()


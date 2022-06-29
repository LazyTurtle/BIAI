import logging
from datetime import datetime
from src.resource import ResourceModel
from src.resource import Collector
import neat
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.substrate import Substrate

fit_max = list()
fit_mean = list()


def batches(list_to_batch, batch_size=1):
    list_length = len(list_to_batch)
    for index in range(0, list_length, batch_size):
        end_batch = min(index + batch_size, list_length)
        # logging.info(f"Batch {index}-{end_batch}")
        yield list_to_batch[index:end_batch]


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

    batch_size = 1

    assert num_collectors == len(
        genomes), f"The number of collectors ({num_collectors}) does not match the number of genomes ({len(genomes)})"

    input_coo, hidden_coo, output_coo = Collector.topology()
    substrate = Substrate(input_coo, output_coo, hidden_coo)

    for batch in batches(genomes, batch_size):
        n_agents = len(batch)
        environment = ResourceModel(width, height, n_agents, num_resources, num_gathering_points)
        collectors = environment.agents(Collector)

        for i in range(len(batch)):
            genome_id, genome = batch[i]
            cppn = neat.nn.FeedForwardNetwork.create(genome, config)
            nn = create_phenotype_network(cppn, substrate)
            agent = collectors[i]
            # nn.reset()  # in case we do multiple trials we have to reset the rnn before each of one
            agent.evolution_setup(nn, genome)

        steps_used = 0
        for i in range(steps):
            has_converged = environment.step()
            steps_used = i + 1
            if has_converged:
                break

        for agent in collectors:
            agent.points = agent.points * (steps / steps_used)
            agent.genome.fitness = agent.points

    fitnesses = [g.fitness for _, g in genomes]
    max_fitness = max(fitnesses)
    mean_fitness = sum(fitnesses) / len(fitnesses)
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

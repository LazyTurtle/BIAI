import logging
import neat
import os
import yaml
from datetime import datetime
from src.resource import ResourceModel
from src.resource import Collector
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.substrate import Substrate

fit_max = list()
fit_mean = list()

NEAT_CONFIG_FILE_PATH = "config/High Mutation.config"
EVOLUTION_CONFIG_FILE_PATH = "config/Evolution.yaml"
ECONFIG = None


def evolve(genomes, config):
    input_coo, hidden_coo, output_coo = Collector.topology()
    substrate = Substrate(input_coo, output_coo, hidden_coo)
    batch_size = ECONFIG["batch_size"]
    width = ECONFIG["width"]
    height = ECONFIG["height"]
    num_resources = ECONFIG["resources"]
    num_gathering_points = ECONFIG["gathering_points"]
    steps = ECONFIG["steps"]

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
            agent.genome.fitness = agent.points * (steps / steps_used)

    fitnesses = [g.fitness for _, g in genomes]
    max_fitness = max(fitnesses)
    mean_fitness = sum(fitnesses) / len(fitnesses)
    fit_max.append(max_fitness)
    fit_mean.append(mean_fitness)
    logging.info(f"Generation {len(fit_max)}")
    logging.info(f"Mean fitness: {mean_fitness}")
    logging.info(f"Max fitness: {max_fitness}")


def batches(list_to_batch, batch_size=1):
    list_length = len(list_to_batch)
    for index in range(0, list_length, batch_size):
        end_batch = min(index + batch_size, list_length)
        # logging.info(f"Batch {index}-{end_batch}")
        yield list_to_batch[index:end_batch]


def setup_logging():
    # filename = datetime.now().strftime('logs/log_%H_%M_%d_%m_%Y.log')
    filename = datetime.now().strftime('logs/log_%d_%m_%Y.log')
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        filemode="w+")  # I'm only interested in the last log of the day, so I just overwrite over the same file


def best_agent():
    cwd = os.getcwd()
    neat_config_file = os.path.join(cwd, NEAT_CONFIG_FILE_PATH)
    neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                     neat.DefaultStagnation, neat_config_file)
    pop = neat.population.Population(neat_config)
    generations = ECONFIG["generations"]
    best = pop.run(evolve, generations)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fit_max)
    plt.plot(fit_mean)
    plt.legend(["Max fitness", "Mean fitness"])
    plt.show()
    return best


if __name__ == '__main__':
    setup_logging()
    with open(EVOLUTION_CONFIG_FILE_PATH) as config_file:
        ECONFIG = yaml.safe_load(config_file)
    best_agent()

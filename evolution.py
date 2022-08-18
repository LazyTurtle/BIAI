import logging
from datetime import datetime
from src.resource import ResourceModel
from src.resource import Collector

import yaml
from deep_hyperneat.population import Population  # Population class
from deep_hyperneat.phenomes import FeedForwardCPPN as CPPN  # CPPN class
from deep_hyperneat.decode import decode  # Decoder for CPPN -> Substrate
from deep_hyperneat.visualize import draw_net  # optional, for visualizing networks

fit_max = list()
fit_mean = list()
DHN_CONFIG_FILE_PATH = "config/DHN_config.yaml"
CONFIG = None


def evolve(genomes):
    width = CONFIG["environment"]["width"]
    height = CONFIG["environment"]["height"]
    num_resources = CONFIG["environment"]["num_resources"]
    num_gathering_points = CONFIG["environment"]["num_gathering_points"]
    input_dims = CONFIG["network"]["substrate_input_dims"]
    sheet_dims = CONFIG["network"]["substrate_sheet_dims"]
    output_dims = CONFIG["network"]["substrate_output_dims"]
    steps = CONFIG["evolution"]["steps"]

    for batch in batches(genomes, CONFIG["evolution"]["batch_size"]):
        n_agents = len(batch)
        environment = ResourceModel(width, height, n_agents, num_resources, num_gathering_points)
        collectors = environment.agents(Collector)

        for i in range(len(batch)):
            genome_id, genome = batch[i]
            cppn = CPPN.create(genome)
            substrate = decode(cppn, input_dims, output_dims, sheet_dims)
            agent = collectors[i]
            agent.evolution_setup(substrate, genome)

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
        yield list_to_batch[index:end_batch]


def setup_logging():
    # filename = datetime.now().strftime('logs/log_%H_%M_%d_%m_%Y.log')
    filename = datetime.now().strftime('logs/log_%d_%m_%Y.log')
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        filemode="w+")  # I'm only interested in the last log of the day, so I just overwrite over the same file


def best_agent():
    key = CONFIG["population"]["key"]
    size = CONFIG["population"]["size"]
    elitism = CONFIG["population"]["elitism"]
    generations = CONFIG["evolution"]["generations"]
    fitness_goal = 20

    population = Population(key, size, elitism)
    best_individual = population.run(evolve, fitness_goal, generations)
    best_cppn = CPPN.create(best_individual)
    substrate = decode(
        best_cppn,
        CONFIG["network"]["substrate_input_dims"],
        CONFIG["network"]["substrate_output_dims"],
        CONFIG["network"]["substrate_sheet_dims"])

    draw_net(best_cppn, filename="reports/champion_images/cppn")
    draw_net(substrate, filename="reports/champion_images/substrate")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fit_max)
    plt.plot(fit_mean)
    plt.legend(["Max fitness", "Mean fitness"])
    plt.show()
    return best_individual


if __name__ == '__main__':
    setup_logging()
    with open(DHN_CONFIG_FILE_PATH) as config_file:
        CONFIG = yaml.safe_load(config_file)
    best_agent()

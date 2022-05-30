import numpy as np
from source.resource import ResourceModel
from source.resource import Collector
import neat
from pureples.pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.pureples.shared.substrate import Substrate

mesa_config = dict()


def evolve(genomes, config):
    # TODO glue together the code from pureples and mesa. the input is obtained by the agents and should be the list
    #  defined in input_coordinates, in that order

    steps = mesa_config["steps"]
    width = mesa_config["width"]
    height = mesa_config["height"]
    num_collectors = mesa_config["num_collectors"]
    num_resources = mesa_config["num_resources"]
    num_gathering_points = mesa_config["num_gathering_points"]

    environment = ResourceModel(width, height, num_collectors, num_resources, num_gathering_points)
    input_coo, hidden_coo, output_coo = Collector.topology()
    substrate = Substrate(input_coo, output_coo, hidden_coo)

    for genome_id, genome in genomes:
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        nn = create_phenotype_network(cppn, substrate)
        environment.reset()
        # TODO modify in order to train multiple agents in the same environment
        agent = environment.agents(Collector)[0]
        agent.update_sensors()
        nn.reset()
        total_reward = 0.
        for i in range(steps):
            input_data = agent.get_sensor_data()
            # The inputs are flattened in order to both have a list (required by neat) and to follow the order defined
            # by the coordinates of hyper neat
            input_data = input_data.flatten()
            output = nn.activate(input_data)
            action = np.argmax(output)
            reward, done = environment.step(agent=agent, action=action)
            total_reward += reward
            if done:
                break
        final_reward = total_reward / (i + 1)  # we want to favour faster agents
        genome.fitness = final_reward


if __name__ == '__main__':
    neat_config_file = "source/config/NEAT.config"
    generations = 2
    neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                     neat.DefaultStagnation, neat_config_file)
    pop = neat.population.Population(neat_config)
    best = pop.run(evolve, generations)
    print(best)

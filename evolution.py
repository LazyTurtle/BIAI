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
    steps = 50
    width = 10
    height = 10
    num_collectors = 5
    num_resources = 5
    num_gathering_points = 1

    environment = ResourceModel(width, height, num_collectors, num_resources, num_gathering_points)
    input_coo, hidden_coo, output_coo = Collector.topology()
    substrate = Substrate(input_coo, output_coo, hidden_coo)

    for genome_id, genome in genomes:
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        nn = create_phenotype_network(cppn, substrate)
        environment.reset()
        # TODO modify in order to train multiple agents in the same environment
        collector_agents = environment.agents(Collector)
        nn.reset() #  in case we do multiple trials we have to reset the rnn before each of one
        total_reward = 0.
        agents_rewards = []
        for agent in collector_agents:
            for i in range(steps):
                agent.update_sensors()
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
            agents_rewards.append(total_reward / (i + 1))  # we want to favour faster agents
        genome.fitness = sum(agents_rewards)


if __name__ == '__main__':
    neat_config_file = "source/config/NEAT.config"
    generations = 2
    neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                     neat.DefaultStagnation, neat_config_file)
    pop = neat.population.Population(neat_config)
    best = pop.run(evolve, generations)
    print(best)

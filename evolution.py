from source.server import instantiate_server
from source.resource import ResourceModel
from source.resource import Collector
import neat
from pureples.pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.pureples.shared.substrate import Substrate


def evolve(genomes, neat_config, mesa_config):
    # TODO glue together the code from pureples and mesa. the input is obtained by the agents and should be the list defined in input_coordinates, in that order

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
        cppn = neat.nn.FeedForwardNetwork.create(genome, neat_config)
        nn = create_phenotype_network(cppn, substrate)
        fitnesses = list()
        environment.reset()
        # TODO modify in order to train multiple agents in the same environment
        agent = environment.agents(Collector)[0]
        agent.update_sensors()
        nn.reset()
        for _ in range(steps):
            input_data = agent.get_sensor_data()
            # The inputs are flatten in order to both have a list, required by neat, and to follow the order defined by the coordinates of hyper neat
            input_data = input_data.flatten()
            output = nn.activate(input)

            action = np.argmax(o)
            ob, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        fitnesses.append(total_reward)


if __name__ == '__main__':
    neat_config_file = "source/config/NEAT.config"
    generations = 2
    con = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                             neat.DefaultStagnation, neat_config_file)
    pop = neat.population.Population(con)
    pop.run(evolve, generations)

from source.server import instantiate_server
from source.resource import ResourceModel
import neat
from pureples.pureples.hyperneat.hyperneat import create_phenotype_network
def evolve(genomes, config):
    for genome_id, genome in genomes:

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        nn = create_phenotype_network(cppn,substrate)
        fitnesses = list()
        # TODO glue together the code from pureples and mesa. the input is obtained by the agents and should be the list defined in input_coordinates, in that order
        environment.reset()
        nn.reset()
        for _ in range(steps):

            output = net.activate(input)

            action = np.argmax(o)
            ob, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        fitnesses.append(total_reward)




if __name__ == '__main__':
    neat_config_file = "source/config/NEAT.config"
    generations = 2
    con = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation, neat_config_file)
    pop = neat.population.Population(con)
    pop.run(evolve,generations)

"""Using the specified search space and fitness function defined in 'Segmentors' this runs
 the genetic algorithm over that space. Best individuals are stored in the hall of fame (hof)."""

import random
import copy

import json
import inspect
import logging

import deap
from deap import base
from deap import tools
from deap import creator
from scoop import futures

from see import Segmentors
# import Segmentors
# from see import Segmentors_MinParams as Segmentors
# from see import Segmentors_OrgAndReducedParams as Segmentors

def print_best_algorithm_code(individual):
    """Print usable code to run segmentation algorithm based on an
     individual's genetic representation vector."""
    ind_algo = Segmentors.algoFromParams(individual)
    original_function = inspect.getsource(ind_algo.evaluate)
    function_contents = original_function[original_function.find('        '):\
                            original_function.find('return')]
    while function_contents.find('self.params') != -1:
        # print(function_contents[function_contents.find('self.params') +
        #     13:function_contents.find(']')-1])
        function_contents = function_contents.replace(
            function_contents[function_contents.find('self.params'):function_contents.find(']')+1],\
            str(ind_algo.params[function_contents[function_contents.find('self.params') + 13:\
            function_contents.find(']')-1]]))
    function_contents = function_contents.replace('        ', '')
    function_contents = function_contents[function_contents.find('\n\"\"\"')+5:]
    print(function_contents)
    return function_contents

def twoPointCopy(np1, np2):
    """Execute a crossover between two numpy arrays of the same length."""
    assert len(np1) == len(np2)
    size = len(np1)
    point1 = random.randint(1, size)
    point2 = random.randint(1, size - 1)
    if point2 >= point1:
        point2 += 1
    else:  # Swap the two points
        point1, point2 = point2, point1
    np1[point1:point2], np2[point1:point2] = np2[point1:point2].copy(
    ), np1[point1:point2].copy()
    return np1, np2

def skimageCrossRandom(np1, np2, seed=False):
    """Execute a crossover between two arrays (np1 and np2) picking a random
     amount of indexes to change between the two."""
    # DO: Only change values associated with algorithm
    assert len(np1) == len(np2)
    # The number of places that we'll cross
    crosses = random.randrange(len(np1))
    # We pick that many crossing points
    indexes = random.sample(range(0, len(np1)), crosses)
    # And at those crossing points, we switch the parameters

    for i in indexes:
        np1[i], np2[i] = np2[i], np1[i]

    return np1, np2


def mutate(copy_child, pos_vals, flip_prob=0.5, seed=False):
    """Change a few of the parameters of the weighting a random number against the flip_prob.

    Keyword arguments:
    copy_child -- the individual to mutate.
    pos_vals -- list of lists where each list are the possible
                values for that particular parameter.
    flip_prob -- how likely it is that we will mutate each value.
                It is computed seperately for each value.

    Outputs:
    child -- New, possibly mutated, individual.

    """
    # Just because we chose to mutate a value doesn't mean we mutate
    # Every aspect of the value
    child = copy.deepcopy(copy_child)

    # Not every algorithm is associated with every value
    # Let's first see if we change the algorithm
    rand_val = random.random()
    if rand_val < flip_prob:
        # Let's mutate
        child[0] = random.choice(pos_vals[0])
    # Now let's get the indexes (parameters) related to that value
    #switcher = AlgoHelp().algoIndexes()
    #indexes = switcher.get(child[0])

    for index in enumerate(pos_vals):
        rand_val = random.random()
        if rand_val < flip_prob:
            # Then we mutate said value
            if index == 22:
                # Do some special
                my_x = random.choice(pos_vals[22])
                my_y = random.choice(pos_vals[23])
                my_z = random.choice(pos_vals[24])
                child[index] = (my_x, my_y, my_z)
                continue
            child[index] = random.choice(pos_vals[index])
    return child


# DO: Make a toolbox from a list of individuals
# DO: Save a population as a list of indivudals (with fitness functions?)
def makeToolbox(pop_size):
    """Make a genetic algorithm toolbox using DEAP. The toolbox uses premade functions
     for crossover, mutation, evaluation and fitness.

    Keyword arguments:
    pop_size -- The size of our population, or how many individuals we have

    """
    # Minimizing fitness function
    creator.create("FitnessMin", base.Fitness, weights=(-0.000001,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # The functions that the GA knows
    toolbox = base.Toolbox()

    # Genetic functions
    toolbox.register("mate", skimageCrossRandom)  # crossover
    toolbox.register("mutate", mutate)  # Mutation
    toolbox.register("evaluate", Segmentors.runAlgo)  # Fitness
    toolbox.register("select", tools.selTournament, tournsize=5)  # Selection
    toolbox.register("map", futures.map)  # So that we can use scoop

    # DO: May want to later do a different selection process

    # We choose the parameters, for the most part, random
    params = Segmentors.parameters()

    for key in params.pkeys:
        toolbox.register(key, random.choice, eval(params.ranges[key]))

    func_seq = []
    for key in params.pkeys:
        func_seq.append(getattr(toolbox, key))

    # Here we populate our individual with all of the parameters
    toolbox.register("individual", tools.initCycle,
                     creator.Individual, func_seq, n=1)

    # And we make our population
    toolbox.register("population", tools.initRepeat,
                     list, toolbox.individual, n=pop_size)

    return toolbox


def initIndividual(icls, content):
    """Create a new individual."""
    logging.getLogger().info(f"In initIndividual={content}")
    return icls(content)


def initPopulation(pcls, ind_init, filename):
    """Create a population by initializing our specified number of individuals."""
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)


class Evolver(object):
    """Perform the genetic algorithm by initializing a population and evolving it over a
     specified number of generations to find the optimal algorithm and parameters for the problem.

    Functions:
    newpopulation -- Initialize a new population.
    writepop -- Records our population in the file "filename".
    readpop -- Reads in existing population from "filename".
    popfitness -- Calculates the fitness values for our population.
    mutate -- Performs mutation and crossover on population.
    nextgen -- Generates the next generation of our population.
    run -- Runs the genetic algorithm.

    """

    AllVals = []
    my_p = Segmentors.parameters()
    for key in my_p.pkeys:
        AllVals.append(eval(my_p.ranges[key]))

    def __init__(self, img, mask, pop_size=10):
        """Set default values for the variables.

        Keyword arguments:
        img -- The original training image
        mask -- The ground truth segmentation mask for the img
        pop_size -- Integer value denoting size of our population,
            or how many individuals there are (default 10)

        """
        # Build Population based on size
        self.img = img
        self.mask = mask
        self.tool = makeToolbox(pop_size)
        self.hof = deap.tools.HallOfFame(10)
        self.best_avgs = []
        self.gen = 0
        self.cxpb, self.mutpb, self.flip_prob = 0.9, 0.9, 0.9

    def newpopulation(self):
        """Initialize a new population."""
        return self.tool.population()

    def writepop(self, tpop, filename='test.json'):
        """Record the population in the file "filename".

        Keyword arguments:
        tpop -- The population to be recorded.
        filename -- string denoting file in which to record
            the population. (default 'test.json')

        """
        logging.getLogger().info(f"Writting population to {filename}")
        with open(filename, 'w') as outfile:
            json.dump(tpop, outfile)

    def readpop(self, filename='test.json'):
        """Read in existing population from "filename"."""
        logging.getLogger().info(f"Reading population from {filename}")
        self.tool.register("population_read", initPopulation,
                           list, creator.Individual, filename)

        self.tool.register("individual_guess",
                           initIndividual, creator.Individual)
        self.tool.register("population_guess", initPopulation,
                           list, self.tool.individual_guess, "my_guess.json")

        return self.tool.population_read()

    def popfitness(self, tpop):
        """Calculate the fitness values for the population, and log general statistics about these
         values. Uses hall of fame (hof) to keep track of top 10 individuals.

        Keyword arguments:
        tpop -- current population

        Outputs:
        extract_fits -- Fitness values for our population
        tpop -- current population

        """
        new_image = [self.img for i in range(0, len(tpop))]
        new_val = [self.mask for i in range(0, len(tpop))]
        fitnesses = map(self.tool.evaluate, new_image, new_val, tpop)

        # DO: Dirk is not sure exactly why we need these
        for ind, fit in zip(tpop, fitnesses):
            ind.fitness.values = fit
        extract_fits = [ind.fitness.values[0] for ind in tpop]

        self.hof.update(tpop)

        #Algo = AlgorithmSpace(AlgoParams)

        # Evaluating the new population
        leng = len(tpop)
        mean = sum(extract_fits) / leng
        self.best_avgs.append(mean)
        sum1 = sum(i*i for i in extract_fits)
        stdev = abs(sum1 / leng - mean ** 2) ** 0.5
        logging.getLogger().info(f"Generation: {self.gen}")
        logging.getLogger().info(f" Min: {min(extract_fits)}")
        logging.getLogger().info(f" Max: {max(extract_fits)}")
        logging.getLogger().info(f" Avg: {mean}")
        logging.getLogger().info(f" Std: {stdev}")
        logging.getLogger().info(f" Size: {leng}")
        #logging.info(" Time: ", time.time() - initTime)
        logging.getLogger().info(f"Best Fitness: {self.hof[0].fitness.values}")
        logging.getLogger().info(f"{self.hof[0]}")
        # Did we improve the population?
        # past_pop = tpop
        # past_min = min(extract_fits)
        # past_mean = mean

        self.gen += self.gen

        return extract_fits, tpop

    def mutate(self, tpop):
        """Return new population with mutated individuals. Perform both mutation and crossover.

        Keyword arguments:
        tpop -- current population

        Output:
        final -- new population with mutated individuals.

        """
        # Calculate next population

        my_sz = len(tpop)
        top = 0  # round(0.1 * my_sz)
        var = round(0.4 * my_sz)
        ran = my_sz - top - var

        offspring = self.tool.select(tpop, var)
        offspring = list(map(self.tool.clone, offspring))  # original code

        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # Do we crossover?
            if random.random() < self.cxpb:
                self.tool.mate(child1, child2)
                # The parents may be okay values so we should keep them
                # in the set
                del child1.fitness.values
                del child2.fitness.values

        # mutation
        for mutant in offspring:
            if random.random() < self.mutpb:
                self.tool.mutate(mutant, self.AllVals, self.flip_prob)
                del mutant.fitness.values

        # new
        pop = self.tool.population()

        final = offspring + pop[0:ran]

        # Replacing the old population
        return final

    def nextgen(self, tpop):
        """Generate the next generation of the population.

        Keyword arguments:
        tpop -- current population

        """
        _, tpop = self.popfitness(tpop)
        return self.mutate(tpop)

    def run(self, ngen=10, startfile=None, checkpoint=None):
        """Run the genetic algorithm, updating the population over ngen number of generations.

        Keywork arguments:
        ngen -- number of generations to run the genetic algorithm.
        startfile -- File containing existing population (default None)
        checkpoint -- File containing existing checkpoint (default None)

        Output:
        population -- Resulting population after ngen generations.

        """
        if startfile:
            population = self.readpop(startfile)
        else:
            population = self.newpopulation()
            if checkpoint:
                self.writepop(population, filename=f"0_{checkpoint}")
        for cur_g in range(1, ngen+1):
            print(f"generation {cur_g} of population size {len(population)}")
            population = self.nextgen(population)

            seg = Segmentors.algoFromParams(self.hof[0])
            mask = seg.evaluate(self.img)
            fitness, _ = Segmentors.FitnessFunction(self.mask, mask)
            print(f"#BEST - {fitness} - {self.hof[0]}")

            if checkpoint:
                print(f"Writing Checkpoint file - {checkpoint}")
                self.writepop(population, filename=f"{checkpoint}_{cur_g}")
                for cur_p in enumerate(population):
                    logging.getLogger().info(population[cur_p])
        return population

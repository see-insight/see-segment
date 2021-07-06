"""Using the specified search space and fitness function defined in 'Algorithm'. 
This runs
the genetic algorithm over that space. Best individuals are stored in the hall of fame (hof).
"""

import random
import copy

import json
import logging
from shutil import copyfile
from pathlib import Path

import deap
from deap import base
from deap import tools
from deap import creator
from scoop import futures

from see import base_classes


# TODO Change algoirthm and algo_instance to be more clear.  use
# consistant naming.


def twoPointCopy(np1, np2, seed=False):
    """Execute a crossover between two numpy arrays of the same length."""
    #if seed:
    #    random.seed(0)
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
    """Execute a crossover.
    
    Between two arrays (np1 and np2) picking a random
    amount of indexes to change between the two.
    """
    #if seed == True:
    #    random.seed(0)
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

    for index in range(len(pos_vals)):
        rand_val = random.random()
        if rand_val < flip_prob:
            #             # Then we mutate said value
            #             if index == 22:
            #                 # Do some special
            #                 my_x = random.choice(pos_vals[22])
            #                 my_y = random.choice(pos_vals[23])
            #                 my_z = random.choice(pos_vals[24])
            #                 child[index] = (my_x, my_y, my_z)
            #                 continue
            child[index] = random.choice(pos_vals[index])
    return child


# DO: Make a toolbox from a list of individuals
# DO: Save a population as a list of indivudals (with fitness functions?)

# TODO: change algo_instance to an algorithm class.
def makeToolbox(pop_size, algo_constructor):
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
    # toolbox.register("mutate", mutate)  # Mutation
    toolbox.register("mutate", base_classes.mutateAlgo)  # Mutation
    toolbox.register("evaluate", algo_constructor.runAlgo)  # Fitness
    toolbox.register("select", tools.selTournament, tournsize=5)  # Selection
    toolbox.register("map", futures.map)  # So that we can use scoop

    # DO: May want to later do a different selection process

    # We choose the parameters, for the most part, random
    algo_instance = algo_constructor()
    params = algo_instance.params

    for key in params.pkeys:
        toolbox.register(key, random.choice, params.ranges[key])

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


##### FILE I/O #####
# TODO Think about moving this to another file?

# TODO make it so we can read from json, pickle or text.
def write_algo_vector(fpop_file, outstring):
    """Write Text output"""
    print(f"Writing in {fpop_file}")
    with open(fpop_file, 'a') as myfile:
        myfile.write(f'{outstring}\n')


def read_algo_vector(fpop_file):
    """Read Text output"""
    print(f"Reading in {fpop_file}")
    inlist = []
    with open(fpop_file, 'r') as myfile:
        for line in myfile:
            inlist.append(eval(line))
    return inlist


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

#     AllVals = []
# #     my_p=param_space
#     for key in my_p.pkeys:
#         AllVals.append(my_p.ranges[key])

    def __init__(self, algo_constructor, data, pop_size=10):
        """Set default values for the variables.

        Keyword arguments:
        img -- The original training image
        mask -- The ground truth segmentation mask for the img
        pop_size -- Integer value denoting size of our population,
            or how many individuals there are (default 10)

        """
        # Build Population based on size
        self.data = data
        self.algo_constructor = algo_constructor
        self.tool = makeToolbox(pop_size, algo_constructor)
        self.hof = deap.tools.HallOfFame(10)
        self.best_avgs = []
        self.gen = 0
        self.cxpb, self.mutpb, self.flip_prob = 0.9, 0.9, 0.9
        
    #TODO add some checking to make sure lists are the right size and type
    #TODO think about how we want to add in fitness to these?
    def copy_individual(self,fromlist):
        """Return individual from list of individuals"""
        new_individual = self.tool.individual()
        for index in range(len(new_individual)):
            new_individual[index] = fromlist[index]
        return new_individual

    # TODO add some checking (see next comment)
    def copy_pop_list(self, tpop):
        """Copy population list to new list"""
        new_tpop = []
        for individual in tpop:
            new_tpop.append(self.copy_individual(individual))
        return new_tpop

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
        filen = Path(filename)

        if filen.suffix == ".txt":
            list_of_lists = read_algo_vector(filen)
            tpop = self.copy_pop_list(list_of_lists)
            return tpop

        logging.getLogger().info(f"Reading population from {filename}")
        self.tool.register("population_read", initPopulation,
                           list, creator.Individual, filename)

        self.tool.register("individual_guess",
                           initIndividual, creator.Individual)

        self.tool.register("population_guess", initPopulation,
                           list, self.tool.individual_guess, "my_guess.json")

        return self.tool.population_read()

    def popfitness(self, tpop):
        """Calculate the fitness values for the population.
        
        Also,log general statistics about these
        values. Uses hall of fame (hof) to keep track of top 10 individuals.

        Keyword arguments:
        tpop -- current population

        Outputs:
        extract_fits -- Fitness values for our population
        tpop -- current population

        """
        # make copies of self.data
        data_references = [copy.deepcopy(self.data)
                           for i in range(0, len(tpop))]
        algos = [self.algo_constructor(paramlist=list(ind)) for ind in tpop]

        # Map the evaluation command to reference data and then to population
        # list
        outdata = map(self.tool.evaluate, algos, data_references)

        # Loop though outputs and add them to ind.fitness so we have a complete
        # record.
        for ind, data in zip(tpop, outdata):
            print(f"fitness={data.fitness}\n")
            ind.fitness.values = [data.fitness]
        extract_fits = [ind.fitness.values[0] for ind in tpop]

        self.hof.update(tpop)

        #Algo = AlgorithmSpace(AlgoParams)

        # Evaluating the new population
        leng = len(tpop)
        mean = sum(extract_fits) / leng
        self.best_avgs.append(mean)
        sum1 = sum(i * i for i in extract_fits)
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

    def mutate(self, tpop, keep_prob=0.1, mutate_prob=0.4):
        """Return new population with mutated individuals. Perform both mutation and crossover.

        Keyword arguments:
        tpop -- current population

        Output:
        final -- new population with mutated individuals.

        """
        # Calculate next population

        # TODO: There is an error here. We need to make sure the best hof is
        # included?

        my_sz = len(tpop)  # Length of current population
        top = min(10, max(1, round(keep_prob * my_sz)))
        top = min(top, len(self.hof))
        var = max(1, round(mutate_prob * my_sz))
        var = min(var, len(self.hof))
        ran = my_sz - top - var

#         print(f"pop[0:{top}:{var}:{ran}]")
#         print(f"pop[0:{top}:{top+var}:{my_sz}]")

#         offspring = self.tool.select(tpop, var)
#         offspring = list(map(self.tool.clone, offspring))  # original code

        offspring = copy.deepcopy(list(self.hof))

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
                self.tool.mutate(self.algo_constructor, mutant, self.flip_prob)
                del mutant.fitness.values

        # new
        #population = self.newpopulation()
        pop = self.tool.population()

        final = pop[0:ran]
        #print(f"pop size should be {len(final)}")
        final += self.hof[0:top]
        #print(f"pop size should be {len(final)}")
        final += offspring[0:var]
        #print(f"pop size should be {len(final)}")

        # print(f"pop[0:{top}:{var}:{ran}]")
        #print(f"pop size should be {len(final)}")

        # Replacing the old population
        return final

    def nextgen(self, tpop):
        """Generate the next generation of the population.

        Keyword arguments:
        tpop -- current population

        """
        _, tpop = self.popfitness(tpop)
        return self.mutate(tpop)

    def run(
            self,
            ngen=10,
            population=None,
            startfile=None,
            checkpoint=None,
            cp_freq=1):
        """Run the genetic algorithm, updating the population over ngen number of generations.
        
        Keywork arguments:
        ngen -- number of generations to run the genetic algorithm.
        startfile -- File containing existing population (default None)
        checkpoint -- File containing existing checkpoint (default None)

        Output:
        population -- Resulting population after ngen generations.
        """
        if startfile:
            try:
                print(f"Reading in {startfile}")
                population = self.readpop(startfile)
            except FileNotFoundError:
                print("WARNING: Start file not found")
            except BaseException:
                raise

        if not population:
            print(f"Initializing a new random population")
            population = self.newpopulation()
            if checkpoint:
                self.writepop(population, filename=f"{checkpoint}")
        else:
            print(f"Using existing population")

        for cur_g in range(0, ngen+1):
            print(f"Generation {cur_g}/{ngen} of population size {len(population)}")

            _, population = self.popfitness(population)

            bestsofar = self.hof[0]

            # Create a new instance from the current algorithm
            # seg = self.algo_constructor(bestsofar)
            # self.data = seg.pipe(self.data)
            fitness = bestsofar.fitness.values[0]
            print(f"#BEST [{fitness},  {bestsofar}]")

            if checkpoint and cur_g % cp_freq == 0:
                print(f"Writing Checkpoint file - {checkpoint}")
                copyfile(f"{checkpoint}", f"{checkpoint}.prev")
                self.writepop(population, filename=f"{checkpoint}")
                for cur_p in range(len(population)):
                    logging.getLogger().info(population[cur_p])

            if cur_g < ngen+1:
                if bestsofar.fitness.values[0] >= 0.95:
                    print("Bestsofar not good enough (>=0.95) restarting population")
                    population = self.newpopulation()
                  # if the best fitness value is at or above the
                  # threshold of 0.95, discard the entire current
                  # population and randomly select a new population
                  # for the next generation
                  # note: setting keep_prob = 0 and mutate_prob = 1
                  # as mutate arguments
                  # should have same result as self.new_population()
                else:
                    print("Mutating Population")
                    population = self.mutate(population)
                  # if the best fitness value is below this threshold,
                  # proceed as normal, mutating the current population
                  # to get the next generation

        if checkpoint:
            print(f"Writing Checkpoint file - {checkpoint}")
            copyfile(f"{checkpoint}", f"{checkpoint}.prev")
            self.writepop(population, filename=f"{checkpoint}")
            for cur_p in range(len(population)):
                logging.getLogger().info(population[cur_p])
        return population

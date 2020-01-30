import random
import copy

import json

import deap
from deap import base
from deap import tools
from deap import creator
from scoop import futures
import logging
import inspect

from see import Segmentors

''' prints usable code to run segmentation algorithm based on an individuals
genetic representation vector
'''
def printBestAlgorithmCode(individual):
    ind_algo = Segmentors.algoFromParams(individual)
    original_function = inspect.getsource(ind_algo.evaluate)
    function_contents = original_function[original_function.find('        '):original_function.find('return')]
    while function_contents.find('self.params') != -1:
        print(function_contents[function_contents.find('self.params') + 
            13:function_contents.find(']')-1])
        function_contents = function_contents.replace(
            function_contents[function_contents.find('self.params'):function_contents.find(']')+1], 
            str(ind_algo.params[function_contents[function_contents.find('self.params') + 
            13:function_contents.find(']')-1]]))
    function_contents = function_contents.replace('        ', '')
    print(function_contents)

'''Executes a crossover between two numpy arrays of the same length '''
def twoPointCopy(np1, np2):
    assert(len(np1) == len(np2))
    size = len(np1)
    point1 = random.randint(1, size)
    point2 = random.randint(1, size-1)
    if (point2 >= point1):
        point2 += 1
    else:  # Swap the two points
        point1, point2 = point2, point1
    np1[point1:point2], np2[point1:point2] = np2[point1:point2].copy(
    ), np1[point1:point2].copy()
    return np1, np2


'''Executes a crossover between two arrays (np1 and np2) picking a
random amount of indexes to change between the two.
'''
def skimageCrossRandom(np1, np2):
    # TODO: Only change values associated with algorithm
    assert(len(np1) == len(np2))
    # The number of places that we'll cross
    crosses = random.randrange(len(np1))
    # We pick that many crossing points
    indexes = random.sample(range(0, len(np1)), crosses)
    # And at those crossing points, we switch the parameters

    for i in indexes:
        np1[i], np2[i] = np2[i], np1[i]

    return np1, np2


''' Changes a few of the parameters of the weighting a random
    number against the flipProb.
    Variables:
    copyChild is the individual to mutate
    posVals is a list of lists where each list are the possible
        values for that particular parameter
    flipProb is how likely, it is that we will mutate each value.
        It is computed seperately for each value.
'''


def mutate(copyChild, posVals, flipProb=0.5):

    # Just because we chose to mutate a value doesn't mean we mutate
    # Every aspect of the value
    child = copy.deepcopy(copyChild)

    # Not every algorithm is associated with every value
    # Let's first see if we change the algorithm
    randVal = random.random()
    if randVal < flipProb:
        # Let's mutate
        child[0] = random.choice(posVals[0])
    # Now let's get the indexes (parameters) related to that value
    #switcher = AlgoHelp().algoIndexes()
    #indexes = switcher.get(child[0])

    for index in range(len(posVals)):
        randVal = random.random()
        if randVal < flipProb:
            # Then we mutate said value
            if index == 22:
                # Do some special
                X = random.choice(posVals[22])
                Y = random.choice(posVals[23])
                Z = random.choice(posVals[24])
                child[index] = (X, Y, Z)
                continue
            child[index] = random.choice(posVals[index])
    return child


# TODO Make a toolbox from a list of individuals
# TODO Save a population as a list of indivudals (with fitness functions?)
def makeToolbox(pop_size):

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

    # TODO: May want to later do a different selection process

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
    logging.getLogger().info(f"In initIndividual={content}")
    return icls(content)


def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)


class Evolver(object):

    AllVals = []
    p = Segmentors.parameters()
    for key in p.pkeys:
        AllVals.append(eval(p.ranges[key]))

    def __init__(self, img, mask, pop_size=10):
        # Build Population based on size
        self.img = img
        self.mask = mask
        self.tool = makeToolbox(pop_size)
        self.hof = deap.tools.HallOfFame(10)
        self.BestAvgs = []
        self.gen = 0
        self.cxpb, self.mutpb, self.flipProb = 0.9, 0.9, 0.9

    def newpopulation(self):
        return self.tool.population()

    def writepop(self, tpop, filename='test.json'):
        logging.getLogger().info(f"Writting population to {filename}")
        with open(filename, 'w') as outfile:
            json.dump(tpop, outfile)

    def readpop(self, filename='test.json'):
        logging.getLogger().info(f"Reading population from {filename}")
        self.tool.register("population_read", initPopulation,
                           list, creator.Individual, filename)

        self.tool.register("individual_guess",
                           initIndividual, creator.Individual)
        self.tool.register("population_guess", initPopulation,
                           list, self.tool.individual_guess, "my_guess.json")

        return self.tool.population_read()

    def popfitness(self, tpop):
        NewImage = [self.img for i in range(0, len(tpop))]
        NewVal = [self.mask for i in range(0, len(tpop))]
        fitnesses = map(self.tool.evaluate, NewImage, NewVal, tpop)

        # TODO: Dirk is not sure exactly why we need these
        for ind, fit in zip(tpop, fitnesses):
            ind.fitness.values = fit
        extractFits = [ind.fitness.values[0] for ind in tpop]

        self.hof.update(tpop)

        #Algo = AlgorithmSpace(AlgoParams)

        # Evaluating the new population
        leng = len(tpop)
        mean = sum(extractFits) / leng
        self.BestAvgs.append(mean)
        sum1 = sum(i*i for i in extractFits)
        stdev = abs(sum1 / leng - mean ** 2) ** 0.5
        logging.getLogger().info(f"Generation: {self.gen}")
        logging.getLogger().info(f" Min: {min(extractFits)}")
        logging.getLogger().info(f" Max: {max(extractFits)}")
        logging.getLogger().info(f" Avg: {mean}")
        logging.getLogger().info(f" Std: {stdev}")
        logging.getLogger().info(f" Size: {leng}")
        #logging.info(" Time: ", time.time() - initTime)
        logging.getLogger().info(f"Best Fitness: {self.hof[0].fitness.values}")
        logging.getLogger().info(f"{self.hof[0]}")
        # Did we improve the population?
        pastPop = tpop
        pastMin = min(extractFits)
        pastMean = mean

        self.gen += self.gen

        return extractFits, tpop

    def mutate(self, tpop):
        # Calculate next population

        sz = len(tpop)
        top = 0  # round(0.1 * sz)
        var = round(0.4 * sz)
        ran = sz - top - var

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
                self.tool.mutate(mutant, self.AllVals, self.flipProb)
                del mutant.fitness.values

        # new
        pop = self.tool.population()

        final = offspring + pop[0:ran]

        # Replacing the old population
        return final

    def nextgen(self, tpop):
        fitness, tpop = self.popfitness(tpop)
        return self.mutate(tpop)

    def run(self, ngen=10, startfile=None, checkpoint=None):
        if startfile:
            population = self.readpop(startfile)
        else:
            population = self.newpopulation()
            if checkpoint:
                self.writepop(population, filename=f"0_{checkpoint}")
        for g in range(1, ngen+1):
            population = self.nextgen(population)
            if checkpoint:
                self.writepop(population, filename=f"{g}_{checkpoint}")
                for p in range(len(population)):
                    logging.getLogger().info(population[p])
        return population


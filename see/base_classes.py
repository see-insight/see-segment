"""The base_classes module is used for the rest of the image grammar to set up base classes."""
import copy
import time
import random
import inspect


class pipedata(object):
    """The pipedata is just an instance of a basic python object. It is used to dynamically
    store output data from a wide variety of algorithms. Most algorithms in the pipe jsut add
    data to this objet which is passed in as an input argument and returned as an output argument.
    """
    pass


class param_space(dict):
    """Construct an parameter dictionary that represents the search space.

    Components:
        pkeys - paramters keys used by the current algorithsm.
        descriptions - Descriptions of the parameters
        ranges - List of possible choices for each parameter.
    """

    descriptions = dict()
    ranges = dict()
    pkeys = []

    @classmethod
    def add(cls, key, prange, description):
        """This is a class function which adds in parameters.

        Inputs:
            key - the paramter name
            prange - the parameter range
            description - the description of the parameter
        """
        cls.descriptions[key] = description
        cls.ranges[key] = prange
        if key not in cls.pkeys:
            cls.pkeys.append(key)

    def addall(self, params):
        """Function to add a list of paramters to the current paramter list"""
        if issubclass(type(params), param_space):
            for key in params:
                self.add(key, params.ranges[key], params.descriptions[key])
                self[key] = params[key]
        else:
            raise TypeError('A very specific bad thing happened.')

    def printparam(self, key):
        """Return description of parameter from param list."""

        outstring = f"{key}={self[key]}\n\t{self.descriptions[key]}"

        if len(self.ranges) < 10:
            outstring += "\n\t{self.ranges[key]}\n\n"
        else:
            outstring += "\n\t{self.ranges[key][:2]}...{self.ranges[key][-2:]}\n\n"

        return outstring

#     def __str__(self):
#         """Return descriptions of all parameters in param list."""
#         out = ""
#         for index, k in enumerate(self.pkeys):
#             out += f"{index} " + self.printparam(k)
#         return out

    def tolist(self):
        """Convert dictionary of params into list of parameters."""
        plist = []
        for key in self.pkeys:
            plist.append(self[key])
        return plist

    def fromlist(self, individual):
        """Convert individual's list into dictionary of params."""
        #logging.getLogger().info(f"Parsing Parameter List for {len(individual)} parameters")
        for index, key in enumerate(self.pkeys):
            self[key] = individual[index]


class algorithm(object):
    """Base class for any image alogirthm.

    Functions:
    evaluate -- Run segmentation algorithm to get inferred mask.

    """

    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        self.params = param_space()
        self.set_params(paramlist)

    def set_params(self, paramlist=None):
        if paramlist:
            if issubclass(type(paramlist), param_space):
                self.params = copy.deepcopy(paramlist)
            else:
                # print(f"{type(paramlist)}_paramlist={paramlist}")
                self.params.fromlist(list(paramlist))
        # TODO Comment this back in
        # self.checkparamindex()

    def checkparamindex(self):
        """Check paramiter keys to ensure values are valid"""
        for myparams in self.params.pkeys:
            assert myparams in self.params, f"ERROR {myparams} is not in parameter list"

    def mutateself(self, flip_prob=0.5):
        """Mutate self and return new params."""
        for myparam in self.params.pkeys:
            rand_val = random.random()
            if rand_val < flip_prob:
                self.params[myparam] = random.choice(
                    self.params.ranges[myparam])
        return self.params

    def pipe(self, data):
        """Run segmentation algorithm to get inferred mask."""
        print("WARNING: Default Pipe, doing nothing\n")
        return data

    def __str__(self):
        """Return params for algorithm."""
        mystring = f"{type(self)} parameters: \n"
        for p in self.params.pkeys:
            mystring += f"\t{p} = {self.params[p]}\n"
        return mystring

    def runAlgo(self, data, params=None):
        """Run and evaluate the performance of an individual.

        Keyword arguments:
        data -- pipedata both input and output.
        params -- the list representing an individual in our population

        Output:
        fitness -- resulting fitness value for the individual
        mask -- resulting image mask associated with the individual (if return_mask=True)

        """

        # TODO make this funciton more flexible and allow multiple types of params
        # i'm thinking (list, param_space and algorithm)
        startTime = int(round(time.time() * 1000))
        if params:
            print(f"{seg}")
            seg = type(self)(paramlist=params)
            data = seg.pipe(data)
            endTime = int(round(time.time() * 1000))
            print(f"Time: {(endTime - startTime)/1000} s")
        else:
            print(f"{self}")
            data = self.pipe(data)
            endTime = int(round(time.time() * 1000))
            print(f"Time: {(endTime - startTime)/1000} s")

        return data

    def mutate_self(self, flip_prob=0.5):
        print("using default mutation function")
        for keys in self.params:
            rand_val = random.random()
            if rand_val < flip_prob:
                # Let's mutate the algorithm
                self.params[index] = random.choice(self.params.ranges[index])

    def algorithm_code(self):
        """Print usable code to run segmentation algorithm based on an
         individual's genetic representation vector."""

        original_function = inspect.getsource(self.evaluate)

        return original_function


def mutateAlgo(algorithm, paramlist, flip_prob=0.5):
    """Generate an offspring based on current individual."""
    child = algorithm(paramlist=paramlist)
    child.mutateself(flip_prob=flip_prob)
    return child


def popCounts(pop):
    """Count the number of each algorihtm in a population"""
    algorithms = seg_params.ranges["algorithm"]
    counts = {a: 0 for a in algorithms}
    for p in pop:
        # print(p[0])
        counts[p[0]] += 1
    return counts

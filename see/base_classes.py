class pipedata(object):
    pass


class param_space(dict):
    """Construct an parameter dictionary that represents the search space.

    Functions:
    printparam -- returns description for each parameter
    tolist -- converts dictionary of params into list
    fromlist -- converts individual into dictionary of params
    """
    
    descriptions = dict()
    ranges = dict()
    pkeys = []
    
    @classmethod
    def add(cls, key, prange, description):
        cls.descriptions[key] = description
        cls.ranges[key] = prange
        if not key in cls.pkeys:
            cls.pkeys.append(key)

    def addall(self,params):
        for key in params:
            self.add(key, params.ranges[key], params.descriptions[key])
            self[key] = params[key]
                            
    def printparam(self, key):
        """Return description of parameter from param list."""
        
        #TODO put an if statment to check for len(ranges) < 4
        return f"{key}={self[key]}\n\t{self.descriptions[key]}\n\t{self.ranges[key][:2]}...{self.ranges[key][-2:]}\n\n"

    def __str__(self):
        """Return descriptions of all parameters in param list."""
        out = ""
        for index, k in enumerate(self.pkeys):
            out += f"{index} " + self.printparam(k)
        return out

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
        if paramlist:
            if (type(paramlist) == list):
                self.params.fromlist(paramlist)
            else:
                self.params = paramlist
            
            
    def checkparamindex(self):
#         print(f"pkeys={self.params.pkeys}")
#         print(f"params = {self.params}")
        """Check paramiter index to ensure values are valid"""
        for myparams in self.params.pkeys:
            assert myparams in self.params, f"ERROR {myparams} is not in parameter list"
             
    def mutateself(self, flip_prob=0.5):
        """Mutate self and return new params."""
        for myparam in self.params.pkeys:
            rand_val = random.random()
            if rand_val < flip_prob:
                self.params[myparam] = random.choice(eval(self.params.ranges[myparam]))
        return self.params
    
    def pipe(self, data):
        """Run segmentation algorithm to get inferred mask."""
        return data

    def __str__(self):
        """Return params for algorithm."""
        mystring = f"{self.params['algorithm']} -- \n"
        for p in self.params.pkeys:
            mystring += f"\t{p} = {self.params[p]}\n"
        return mystring


    def runAlgo(self, data, params="none"):
        """Run and evaluate the performance of an individual.

        Keyword arguments:
        img -- training image
        ground_img -- the ground truth for the image mask
        individual -- the list representing an individual in our population
        return_mask -- Boolean value indicating whether to return resulting
         mask for the individual or not (default False)

        Output:
        fitness -- resulting fitness value for the individual
        mask -- resulting image mask associated with the individual (if return_mask=True)

        """
        seg = algorithm(paramlist=params)
        data = seg.pipe(data)
        data.fitness = 1
        return data    
    
    

def algoFromParams(individual):
    """Convert an individual's param list to an algorithm. Assumes order
     defined in the parameters class.

    Keyword arguments:
    individual -- the list representing an individual in our population

    Output:
    algorithm(individual) -- algorithm associated with the individual

    """
    if individual["algorithm"] in segmentor.algorithmspace:
        algorithm = algorithmspace[individual["algorithm"]]
        return algorithm(individual)
    else:
        raise ValueError("Algorithm not avaliable")

    
def mutateAlgo(copy_child, pos_vals, flip_prob=0.5, seed=False):
    """Generate an offspring based on current individual."""

    child = copy.deepcopy(copy_child)
    
    # Not every algorithm is associated with every value
    # Let's first see if we change the algorithm
    rand_val = random.random()
    if rand_val < flip_prob:
        # Let's mutate the algorithm
        child[0] = random.choice(pos_vals[0])

    #use the local search for mutation.
    seg = algoFromParams(child)
    child = seg.mutateself(flip_prob)
    return child


def print_best_algorithm_code(individual):
    """Print usable code to run segmentation algorithm based on an
     individual's genetic representation vector."""
    #ind_algo = Segmentors.algoFromParams(individual)
    ind_algo = algoFromParams(individual)
    original_function = inspect.getsource(ind_algo.evaluate)

    # Get the body of the function
    function_contents = original_function[original_function.find('        '):\
                            original_function.find('return')]
    while function_contents.find('self.params') != -1:

        # Find the index of the 's' at the start of self.params
        params_index = function_contents.find('self.params')

        # Find the index of the ']' at the end of self.params["<SOME_TEXT>"]
        end_bracket_index = function_contents.find(']', params_index)+1

        # Find the first occurance of self.params["<SOME_TEXT>"] and store it
        code_to_replace = function_contents[params_index:end_bracket_index]

        # These offset will be used to access only the params_key
        offset = len('self.params["')
        offset2 = len('"]')

        # Get the params key
        params_key = function_contents[params_index + offset:end_bracket_index-offset2]

        # Use the params_key to access the params_value
        param_value = str(ind_algo.params[params_key])

        # Replace self.params["<SOME_TEXT>"] with the value of self.params["<SOME_TEXT>"]
        function_contents = function_contents.replace(code_to_replace, param_value)

    function_contents = function_contents.replace('        ', '')
    function_contents = function_contents[function_contents.find('\n\"\"\"')+5:]
    print(function_contents)
    return function_contents
   
def popCounts(pop):
    """Count the number of each algorihtm in a population"""
    algorithms = seg_params.ranges["algorithm"]
    counts = {a:0 for a in algorithms}
    for p in pop:
        #print(p[0])
        counts[p[0]] += 1
    return counts
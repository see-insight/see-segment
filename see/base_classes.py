
class param_space(dict):
    """Construct an ordered dictionary that represents the search space.

    Functions:
    printparam -- returns description for each parameter
    tolist -- converts dictionary of params into list
    fromlist -- converts individual into dictionary of params

    """
    
    descriptions = dict()
    ranges = dict()
    pkeys = []
            
    def add(key, prange, description):
        param_space.descriptions[key] = description
        param_space.ranges[key] = prange
        param_space.pkeys.append(key)

    def printparam(self, key):
        """Return description of parameter from param list."""
        return f"{key}={self[key]}\n\t{param_space.descriptions[key]}\n\t{param_space.ranges[key]}\n"

    def __str__(self):
        """Return descriptions of all parameters in param list."""
        out = ""
        for index, k in enumerate(param_space.pkeys):
            out += f"{index} " + param_space.printparam(k)
        return out

    def tolist(self):
        """Convert dictionary of params into list of parameters."""
        plist = []
        for key in param_space.pkeys:
            plist.append(self[key])
        return plist

    def fromlist(self, individual):
        """Convert individual's list into dictionary of params."""
        logging.getLogger().info(f"Parsing Parameter List for {len(individual)} parameters")
        for index, key in enumerate(param_space.pkeys):
            self[key] = individual[index]


class algorithm(object):
    """Base class for any image alogirthm.

    Functions:
    evaluate -- Run segmentation algorithm to get inferred mask.

    """
    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        self.params = paramiter_space()
        if paramlist:
            self.params.fromlist(paramlist)

    def checkparamindex(self):
        """Check paramiter index to ensure values are valid"""
        for myparams in self.paramindexes:
            assert myparams in self.params, f"ERROR {myparams} is not in parameter list"
             
    def mutateself(self, flip_prob=0.5):
        """Mutate self and return new params."""
        for myparam in self.paramindexes:
            rand_val = random.random()
            if rand_val < flip_prob:
                self.params[myparam] = random.choice(eval(self.params.ranges[myparam]))
        return self.params
    
    def evaluate(self, data):
        """Run segmentation algorithm to get inferred mask."""
        return data

    def __str__(self):
        """Return params for algorithm."""
        mystring = f"{self.params['algorithm']} -- \n"
        for p in self.paramindexes:
            mystring += f"\t{p} = {self.params[p]}\n"
        return mystring
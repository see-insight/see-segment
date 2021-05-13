"""The base_classes module is used for the rest of the image grammar to set up base classes."""
import copy
import time    
import random

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
        if not key in cls.pkeys:
            cls.pkeys.append(key)

    def addall(self,params):
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
            if  issubclass(type(paramlist), param_space):
                self.params = copy.deepcopy(paramlist)
            else:
                #print(f"{type(paramlist)}_paramlist={paramlist}")
                self.params.fromlist(list(paramlist))
        #TODO Comment this back in
        #self.checkparamindex()
        
    def checkparamindex(self):
        """Check paramiter keys to ensure values are valid"""
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
        print("WARNING: Default Pipe, doing nothing\n")
        return data

    def __str__(self):
        """Return params for algorithm."""
        mystring = f"{self.params['algorithm']} -- \n"
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

        #TODO make this funciton more flexible and allow multiple types of params 
        # i'm thinking (list, param_space and algorithm)
        startTime = int(round(time.time() * 1000))
        if params:
            seg = type(self)(paramlist=params)
            data = seg.pipe(data)
            endTime = int(round(time.time() * 1000))
            print(f"{seg}Time: {(endTime - startTime)/1000} s")
        else:
            data = self.pipe(data) 
            endTime = int(round(time.time() * 1000))
            print(f"{self}Time: {(endTime - startTime)/1000} s")
        
        return data    
    
def mutateAlgo(copy_child, pos_vals, flip_prob=0.5, seed=False):
    """Generate an offspring based on current individual."""
    
    #print(f"copy_child = {type(copy_child)}")
    child = copy.deepcopy(copy_child)
    
    # Not every algorithm is associated with every value
    # Let's first see if we change the algorithm
    for index,vals in enumerate(pos_vals):
        rand_val = random.random()
        if rand_val < flip_prob:
            # Let's mutate the algorithm
            child[index] = random.choice(pos_vals[index])

#     #use the local search for mutation.
#     seg = algoFromParams(child)
#     child = seg.mutateself(flip_prob)
    return child


def print_best_algorithm_code(individual):
    """Print usable code to run segmentation algorithm based on an
     individual's genetic representation vector."""
    pass

#TODO Try to fix this print. I'm not sure the best way to generate an algorithm from parameters.
#     #ind_algo = Segmentors.algoFromParams(individual)
#     ind_algo = algoFromParams(individual)
#     original_function = inspect.getsource(ind_algo.evaluate)

#     # Get the body of the function
#     function_contents = original_function[original_function.find('        '):\
#                             original_function.find('return')]
#     while function_contents.find('self.params') != -1:

#         # Find the index of the 's' at the start of self.params
#         params_index = function_contents.find('self.params')

#         # Find the index of the ']' at the end of self.params["<SOME_TEXT>"]
#         end_bracket_index = function_contents.find(']', params_index)+1

#         # Find the first occurance of self.params["<SOME_TEXT>"] and store it
#         code_to_replace = function_contents[params_index:end_bracket_index]

#         # These offset will be used to access only the params_key
#         offset = len('self.params["')
#         offset2 = len('"]')

#         # Get the params key
#         params_key = function_contents[params_index + offset:end_bracket_index-offset2]

#         # Use the params_key to access the params_value
#         param_value = str(ind_algo.params[params_key])

#         # Replace self.params["<SOME_TEXT>"] with the value of self.params["<SOME_TEXT>"]
#         function_contents = function_contents.replace(code_to_replace, param_value)

#     function_contents = function_contents.replace('        ', '')
#     function_contents = function_contents[function_contents.find('\n\"\"\"')+5:]
#     print(function_contents)
#     return function_contents
   
def popCounts(pop):
    """Count the number of each algorihtm in a population"""
    algorithms = seg_params.ranges["algorithm"]
    counts = {a:0 for a in algorithms}
    for p in pop:
        #print(p[0])
        counts[p[0]] += 1
    return counts
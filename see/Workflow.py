"""File Workflow.py."""
from see.base_classes import param_space, algorithm


class workflow(algorithm):
    """Class that creates a workflow for a given algorithm."""

    worklist = []

    @classmethod
    def setalgos(cls, algo_list):
        """Resets algorithms to the workflow list."""        
        workflow.worklist = algo_list
    
    @classmethod
    def addalgos(cls, algo_list, append=False):
        """Add algorithms to the workflow list."""
        if len(workflow.worklist) > 0 and append == False:
            raise Exception("ERROR: Workflow already defined. Use append=True to append.")
            
        if isinstance(algo_list, list):
            for algo in algo_list:
                workflow.worklist.append(algo)
        else:
            workflow.worklist.append(algo_list)

    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        self.params = param_space()
        self.set_params(paramlist)
        for algo in workflow.worklist:
            thisalgo = algo()
            self.params.addall(thisalgo.params)
        self.set_params(paramlist)

    def mutateself(self, flip_prob=0.5):
        """Mutate self and return new params."""
        print("using workflow mutate algorithm and looping over workflow")
        for algo in workflow.worklist:
            thisalgo = algo()
            thisalgo.params.addall(thisalgo.params)

            thisalgo.mutateself(flip_prob=flip_prob)
            self.params.addall(thisalgo.params)

    def pipe(self, data):
        """Return parameter data collection for workflow."""
        for algo_constructor in workflow.worklist:
            algo = algo_constructor(self.params)
            algo.params.addall(self.params)
            data = algo.pipe(data)
        return data
    
    def algorithm_code(self):
        output = "# Run a workflow\n"
        for algo in workflow.worklist:
            thisalgo = algo()
            output += thisalgo.algorithm_code()
        return output

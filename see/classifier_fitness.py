import numpy as np

from see.base_classes import algorithm


class ClassifierFitness(algorithm):
    """Contains functions to return result of fitness function.
    and run classifier algorithm
    """

    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        super(ClassifierFitness, self).__init__(paramlist)

    def evaluate(self, predictions, targets):
        #print("predictions", predictions)
        #print("targets", targets)
        length = len(predictions)
        #print("length", length)
        return float(length - np.sum(predictions == targets))/ length

    def pipe(self, data):
        """Run segmentation algorithm to get inferred mask."""
        data.fitness = self.evaluate(data.predictions, data.testing_set.y)
        return data

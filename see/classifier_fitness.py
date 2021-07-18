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
        # print("predictions", predictions)
        # print("targets", targets)
        length = len(predictions)
        # print("length", length)
        return float(length - np.sum(predictions == targets)) / length

    def pipe(self, data):
        """Run segmentation algorithm to get inferred mask."""
        is_data_k_folds = data.k_folds
        if is_data_k_folds:
            testing_folds_y = list(map(lambda fold: fold.y, data.testing_folds))
            fitness_scores = list(map(self.evaluate, data.predictions, testing_folds_y))
            data.fitness = sum(fitness_scores) / len(fitness_scores)
        else:
            data.fitness = self.evaluate(data.predictions, data.testing_set.y)

        return data

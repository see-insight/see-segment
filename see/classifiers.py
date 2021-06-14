"""Classifer algorithm library designed to classify images with a searchable parameter space.
 This libary actually does not incode the search code itself, instead it just defines
  the search parameters and the evaluation funtions. In other words, the following classes
  are wrappers of the scikit-learn classifiers that whose parameters can be used in the
  Genetic Search algorithm."""

import numpy as np
#import sklearn

from sklearn.tree import DecisionTreeClassifier as DecisionTree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as kNearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from see.base_classes import param_space, algorithm, pipedata

class ClassifierParams(param_space):
    """Parameter space for classifiers
    """
    descriptions = dict()
    ranges = dict()


ClassifierParams.add('algorithm',
                     [],
                     "string code for the algorithm")

# TODO: Many of these parameters' ranges were set arbitrarily.
# The max_param were especially arbitrary as some of the documentation
# do not give a suggested range.

ClassifierParams.add("max_iter",
                     [i for i in range(1, 1001)],
                     'Number of iterations for the algorithm')

ClassifierParams.add("alpha",
                     [float(i)/10000 for i in range(1, 10000)],
                     'regularization parameter')

ClassifierParams.add("max_depth",
                     [i for i in range(1, 1001)],
                     'Maximum depth of tree')

ClassifierParams.add("n_estimators",
                     [i for i in range(1, 1001)],
                     'Number of trees in the forest')

ClassifierParams.add("n_neighbors",
                     [i for i in range(1, 1001)],
                     'Number of neighbors to use')


class Classifier(algorithm):
    """Base class for classifier classes defined below.

    Functions:
    evaluate -- Run classifier algorithm to classify dataset.

    """

    algorithmspace = dict()

    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        super().__init__()
        self.params = ClassifierParams()
        self.params["algorithm"] = "MLP Neural Network"
        self.params["max_iter"] = 200
        self.params["alpha"] = 0.0001
        self.params["max_depth"] = 1
        self.params["n_estimators"] = 100
        self.params["n_neighbors"] = 5
        
        self.set_params(paramlist)

    # Input: labelled dataset
    # Output: Tuple with prediction probabilities, score
    def evaluate(self, training_set, testing_set):
        """Instance evaluate method. Needs to be overridden by subclasses."""
        self.thisalgo = Classifier.algorithmspace[self.params['algorithm']](self.params)
        return self.thisalgo.evaluate(training_set, testing_set)

    def pipe(self, data):
        print(data)
        self.thisalgo = Classifier.algorithmspace[self.params['algorithm']](self.params)
        is_data_k_folds = data.k_folds
        if(is_data_k_folds):
            training_folds = data.training_folds
            testing_folds = data.testing_folds
            data.predictions = list(map(self.evaluate, training_folds, testing_folds))
        else:
            data.predictions = self.evaluate(data.training_set, data.testing_set)
        return data

    @classmethod
    def add_classifier(cls, key, classifier):
        """Adds the classifier to the algorithm space."""
        ClassifierParams.ranges['algorithm'].append(key)
        cls.algorithmspace[key] = classifier


class GaussianNBClassifier(Classifier):
    """Perform Gaussian Naive Bayes classification algorithm."""

    def __init__(self, paramlist=None):
        """Gaussian Naive Bayes has no parameters in the example."""
        super().__init__(paramlist)
        self.params["algorithm"] = "Gaussian Naive Bayes"
        self.params = ClassifierParams()

    def evaluate(self, training_set, testing_set):
        """The evaluate function for Gaussian Naive Bayes"""
        print("HHHH RUNNING EVAL FOR GNB")
        clf = GaussianNB()
        clf.fit(training_set.X, training_set.y)
        return clf.predict(testing_set.X)


Classifier.add_classifier('Gaussian Naive Bayes', GaussianNBClassifier)


class KNeighborsClassifier(Classifier):
    """Perform K Nearest-Neighbors classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__()
        self.params = ClassifierParams()
        self.params["algorithm"] = "K Nearest Neighbors"

        self.params["n_neighbors"] = 3
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for K Nearest-Nei3ghbors"""
        print("RUNNING EVAL FOR KNN")

        num_samples = len(training_set.X)
        # TODO n_neighbors must be <= number of samples
        # Modulo might not be the best way to do this.
        neighbors_param = self.params["n_neighbors"]
        print("N NEIGHBORS", neighbors_param)
        param_in_range = (neighbors_param <= num_samples) and (neighbors_param > 0)
        print("PARAM IN RANGE", param_in_range)
        clf = kNearestNeighbors(n_neighbors=(neighbors_param if param_in_range else (neighbors_param % num_samples + 1)))
        clf.fit(training_set.X, training_set.y)
        return clf.predict(testing_set.X)


Classifier.add_classifier('K Nearest Neighbors', KNeighborsClassifier)


class DecisionTreeClassifier(Classifier):
    """Perform Decision Tree classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__()
        self.params = ClassifierParams()

        self.params["algorithm"] = "Decision Tree"
        self.params["max_depth"] = 5
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for Decision Trees."""

        clf = DecisionTree(max_depth=self.params["max_depth"])
        clf.fit(training_set.X, training_set.y)
        return clf.predict(testing_set.X)


Classifier.add_classifier('Decision Tree', DecisionTreeClassifier)


class RandomForestContainer(Classifier):
    """Perform Random Forest classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__()
        self.params = ClassifierParams()

        self.params["algorithm"] = "Random Forest"
        self.params["max_depth"] = 5
        self.params["n_estimators"] = 10
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for Decision Trees."""
        
        clf = RandomForestClassifier(
            max_depth=self.params["max_depth"],
            n_estimators=self.params["n_estimators"])

        clf.fit(training_set.X, training_set.y)
        return clf.predict(testing_set.X)


Classifier.add_classifier('Random Forest', RandomForestContainer)

class MLPContainer(Classifier):
    """Perform MLP Neural Network classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__()
        self.params = ClassifierParams()

        self.params["algorithm"] = "MLP Neural Network"
        self.params["max_iter"] = 1000
        self.params["alpha"] = 1
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for MLP."""

        clf = MLPClassifier(alpha=1, max_iter=1000)

        clf.fit(training_set.X, training_set.y)

        return clf.predict(testing_set.X)


Classifier.add_classifier('MLP Neural Network', MLPContainer)

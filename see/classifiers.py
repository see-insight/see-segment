"""Classifer algorithm library designed to classify images with a searchable parameter space.
 This libary actually does not incode the search code itself, instead it just defines
  the search parameters and the evaluation funtions. In other words, the following classes
  are wrappers of the scikit-learn classifiers that whose parameters can be used in the
  Genetic Search algorithm."""

import numpy as np

# import sklearn

from sklearn.tree import DecisionTreeClassifier as DecisionTree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as kNearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from see.base_classes import param_space, algorithm, pipedata


class ClassifierParams(param_space):
    """Parameter space for classifiers"""

    descriptions = dict()
    ranges = dict()
    pkeys = []
    
    @classmethod
    def empty_space(cls):
        cls.descriptions = dict()
        cls.ranges = dict()
        cls.pkeys = []

    @classmethod
    def use_tutorial_space(cls):
        """Sets parameter space to
        use the tutorial space"""
        # TODO: does nothing; same as default space might not need it...
        cls.empty_space()
        cls.use_default_space()
        return 0

    @classmethod
    def use_default_space(cls):
        
        cls.add("algorithm", [], "string code for the algorithm")

        # TODO: Many of these parameters' ranges were set arbitrarily.
        # The max_param were especially arbitrary as some of the documentation
        # do not give a suggested range.

        cls.add(
            "max_iter", [i for i in range(1, 101)], "Number of iterations for the algorithm"
        )

        cls.add(
            "alpha", [float(i) / 1000 for i in range(1, 1000)], "regularization parameter"
        )

        cls.add("max_depth", [i for i in range(1, 10)], "Maximum depth of tree")

        cls.add(
            "n_estimators", [i for i in range(1, 100)], "Number of trees in the forest"
        )

        cls.add(
            "n_neighbors", [i for i in range(1, 10)], "Number of neighbors to use"
        )

        cls.add(
            "length_scale",
            [float(i) / 10 for i in range(1, 10)],
            "The length scale of the kernel.",
        )

        cls.add(
            "learning_rate", [float(i) / 10 for i in range(1, 10)], "The learning rate"
        )

        cls.add(
            "C", [float(i) / 10 for i in range(1, 20)], "The regularization parameter"
        )

        cls.add( # omitt "precomputed" kernel as human intervention is needed...
            "kernel", ["linear", "poly", "rbf", "sigmoid"], "The kernel for SVC"
        )

        cls.add(
            "gamma",
            ["scale", "auto"],  # Todo: this might need to be a mix of strings and floats....
            "The kernel coefficient for for ‘rbf’, ‘poly’ and ‘sigmoid’ kernels.",
        )

ClassifierParams.use_default_space()

class Classifier(algorithm):
    """Base class for classifier classes defined below.

    Functions:
    evaluate -- Run classifier algorithm to classify dataset.

    """

    algorithmspace = dict()
    # TODO: All Classifiers need the paramindexes field; cannot default to []
    # in order to print properly

    def __init__(self, paramlist=None, paramindexes=[]):
        """Generate algorithm params from parameter list."""
        super().__init__()
        self.params = ClassifierParams()
        self.params["algorithm"] = "MLP Neural Network"
        self.params["max_iter"] = 200
        self.params["alpha"] = 0.0001
        self.params["max_depth"] = 1
        self.params["n_estimators"] = 100
        self.params["n_neighbors"] = 5
        self.params["length_scale"] = 1.0
        self.params["learning_rate"] = 0.1
        self.params["kernel"] = "rbf"
        self.params["C"] = 1
        self.params["gamma"] = "scale"

        self.paramindexes = paramindexes
        self.set_params(paramlist)

    # Input: labelled dataset
    # Output: Tuple with prediction probabilities, score
    def evaluate(self, training_set, testing_set):
        """Instance evaluate method. Needs to be overridden by subclasses."""
        self.thisalgo = Classifier.algorithmspace[self.params["algorithm"]](self.params)
        return self.thisalgo.evaluate(training_set, testing_set)

    def pipe(self, data):
        print(data)
        self.thisalgo = Classifier.algorithmspace[self.params["algorithm"]](self.params)
        is_data_k_folds = data.k_folds
        if is_data_k_folds:
            training_folds = data.training_folds
            testing_folds = data.testing_folds
            data.predictions = list(map(self.evaluate, training_folds, testing_folds))
        else:
            data.predictions = self.evaluate(data.training_set, data.testing_set)
        return data

    @classmethod
    def add_classifier(cls, key, classifier):
        """Adds the classifier to the algorithm space."""
        ClassifierParams.ranges["algorithm"].append(key)
        cls.algorithmspace[key] = classifier

    @classmethod
    def use_tutorial_space(cls):
        """Sets algorithm and parameter space to
        use the tutorial space"""
        # maybe.....this is fine as long as the parameter space tightens???
        # TODO: does nothing
        # TODO: might need to shrink parameterspace to match those
        # needed for the tutorial space...
        ClassifierParams.use_tutorial_space()
        cls.algorithmspace = dict()
        cls.add_classifier("Ada Boost", AdaBoostContainer)
        cls.add_classifier("Decision Tree", DecisionTreeClassifier)
        cls.add_classifier("Gaussian Naive Bayes", GaussianNBClassifier)
        cls.add_classifier("Gaussian Process", GaussianProcessContainer)
        cls.add_classifier("K Nearest Neighbors", KNeighborsClassifier)
        cls.add_classifier("MLP Neural Network", MLPContainer)
        cls.add_classifier("Quadratic Discriminant Analysis", QDAContainer)
        cls.add_classifier("Random Forest", RandomForestContainer)
        cls.add_classifier("SVC", SVCContainer)


class GaussianNBClassifier(Classifier):
    """Perform Gaussian Naive Bayes classification algorithm."""

    def __init__(self, paramlist=None):
        """Gaussian Naive Bayes has no parameters in the example."""

        super(GaussianNBClassifier, self).__init__(paramlist)

        self.params["algorithm"] = "Gaussian Naive Bayes"
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for Gaussian Naive Bayes"""
        clf = GaussianNB()
        clf.fit(training_set.X, training_set.y)
        return clf.predict(testing_set.X)


Classifier.add_classifier("Gaussian Naive Bayes", GaussianNBClassifier)


class KNeighborsClassifier(Classifier):
    """Perform K Nearest-Neighbors classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__()

        self.params["algorithm"] = "K Nearest Neighbors"
        self.params["n_neighbors"] = 3
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for K Nearest-Nei3ghbors"""

        num_samples = len(training_set.X)
        # TODO n_neighbors must be <= number of samples
        # Modulo might not be the best way to do this.
        neighbors_param = self.params["n_neighbors"]
        param_in_range = (neighbors_param <= num_samples) and (neighbors_param > 0)
        clf = kNearestNeighbors(
            n_neighbors=(
                neighbors_param
                if param_in_range
                else (neighbors_param % num_samples + 1)
            )
        )
        clf.fit(training_set.X, training_set.y)
        return clf.predict(testing_set.X)


Classifier.add_classifier("K Nearest Neighbors", KNeighborsClassifier)


class DecisionTreeClassifier(Classifier):
    """Perform Decision Tree classification algorithm."""

    def __init__(self, paramlist=None):
        super(DecisionTreeClassifier, self).__init__()

        self.params["algorithm"] = "Decision Tree"
        self.params["max_depth"] = 5
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for Decision Trees."""

        clf = DecisionTree(max_depth=self.params["max_depth"])
        clf.fit(training_set.X, training_set.y)
        return clf.predict(testing_set.X)


Classifier.add_classifier("Decision Tree", DecisionTreeClassifier)


class RandomForestContainer(Classifier):
    """Perform Random Forest classification algorithm."""

    def __init__(self, paramlist=None):
        super(RandomForestContainer, self).__init__()

        self.params["algorithm"] = "Random Forest"
        self.params["max_depth"] = 5
        self.params["n_estimators"] = 10
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for Decision Trees."""

        clf = RandomForestClassifier(
            max_depth=self.params["max_depth"], n_estimators=self.params["n_estimators"]
        )

        clf.fit(training_set.X, training_set.y)
        return clf.predict(testing_set.X)


Classifier.add_classifier("Random Forest", RandomForestContainer)


class MLPContainer(Classifier):
    """Perform MLP Neural Network classification algorithm."""

    def __init__(self, paramlist=None):
        super(MLPContainer, self).__init__()

        self.params["algorithm"] = "MLP Neural Network"
        self.params["max_iter"] = 1000
        self.params["alpha"] = 1
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for MLP."""

        clf = MLPClassifier(alpha=1, max_iter=1000)

        clf.fit(training_set.X, training_set.y)

        return clf.predict(testing_set.X)


Classifier.add_classifier("MLP Neural Network", MLPContainer)


class GaussianProcessContainer(Classifier):
    """Perform Guassian Process classification algorithm."""

    def __init__(self, paramlist=None):
        super(GaussianProcessContainer, self).__init__()

        self.params["algorithm"] = "Gaussian Process"
        self.params["length_scale"] = 1.0
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for Gaussian Process."""

        clf = GaussianProcessClassifier(1.0 * RBF(self.params["length_scale"]))

        clf.fit(training_set.X, training_set.y)

        return clf.predict(testing_set.X)


Classifier.add_classifier("Gaussian Process", GaussianProcessContainer)


class ExtraTreesContainer(Classifier):
    """Perform Guassian Process classification algorithm."""

    def __init__(self, paramlist=None):
        super(ExtraTreesContainer, self).__init__()

        self.params["algorithm"] = "Extra Trees"
        self.params["n_estimators"] = 100
        self.params["max_depth"] = 5
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for Extra Trees."""

        clf = ExtraTreesClassifier(
            n_estimators=self.params["n_estimators"], max_depth=self.params["max_depth"]
        )

        clf.fit(training_set.X, training_set.y)

        return clf.predict(testing_set.X)


Classifier.add_classifier("Extra Trees", ExtraTreesContainer)


class GradientBoostingContainer(Classifier):
    """Perform Gradient Boosting classification algorithm."""

    def __init__(self, paramlist=None):
        super(GradientBoostingContainer, self).__init__()

        self.params["algorithm"] = "Gradient Boosting"
        self.params["n_estimators"] = 100
        self.params["learning_rate"] = 0.1
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for Gradient Boosting."""

        clf = GradientBoostingClassifier(
            n_estimators=self.params["n_estimators"],
            learning_rate=self.params["learning_rate"],
        )

        clf.fit(training_set.X, training_set.y)

        return clf.predict(testing_set.X)


Classifier.add_classifier("Gradient Boosting", GradientBoostingContainer)


class AdaBoostContainer(Classifier):
    """Perform Ada Boost classification algorithm."""

    def __init__(self, paramlist=None):
        super(AdaBoostContainer, self).__init__()

        self.params["algorithm"] = "Ada Boost"
        self.params["n_estimators"] = 50
        self.params["learning_rate"] = 1
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for Ada Boost."""

        clf = AdaBoostClassifier(
            n_estimators=self.params["n_estimators"],
            learning_rate=self.params["learning_rate"],
        )

        clf.fit(training_set.X, training_set.y)

        return clf.predict(testing_set.X)


Classifier.add_classifier("Ada Boost", AdaBoostContainer)


class SVCContainer(Classifier):
    """Perform SVC classification algorithm."""

    def __init__(self, paramlist=None):
        super(SVCContainer, self).__init__()

        self.params["algorithm"] = "SVC"
        self.params["kernel"] = "rbf"
        self.params["C"] = 1
        self.params["gamma"] = "scale"

        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for SVC Boost."""

        clf = SVC(
            kernel=self.params["kernel"], C=self.params["C"], gamma=self.params["gamma"]
        )

        clf.fit(training_set.X, training_set.y)

        return clf.predict(testing_set.X)


Classifier.add_classifier("SVC", SVCContainer)


class QDAContainer(Classifier):
    """Perform QDA classification algorithm."""

    def __init__(self, paramlist=None):
        super(QDAContainer, self).__init__()

        self.params["algorithm"] = "Quadratic Discriminant Analysis"

        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """The evaluate function for Quadratic Discriminant Analysis."""

        clf = QuadraticDiscriminantAnalysis()

        clf.fit(training_set.X, training_set.y)

        return clf.predict(testing_set.X)


Classifier.add_classifier("Quadratic Discriminant Analysis", QDAContainer)

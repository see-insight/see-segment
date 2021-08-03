"""Classifer algorithm library designed to classify images with a searchable parameter space.
 This libary actually does not incode the search code itself, instead it just defines
  the search parameters and the evaluation funtions. In other words, the following classes
  are wrappers of the scikit-learn classifiers that whose parameters can be used in the
  Genetic Search algorithm."""

import numpy as np

# import sklearn

from sklearn.tree import DecisionTreeClassifier as DecisionTree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

from abc import ABC, abstractmethod

from see.base_classes import param_space, algorithm, pipedata


class ClassifierParams(param_space):
    """Parameter space for classifiers"""

    descriptions = {"algorithm": "string code for the algorithm"}
    ranges = {"algorithm": []}
    pkeys = ["algorithm"]
        
    @classmethod
    def empty_space(cls):
        cls.descriptions = dict()
        cls.ranges = dict()
        cls.pkeys = []
        cls.add("algorithm", [], "string code for the algorithm")

        
    @classmethod
    def get_param(cls, param_name):
        """
        param_name: name of parameter

        returns the ranges, description
        """
        return cls.ranges[param_name], cls.descriptions[param_name]


    @classmethod
    def use_tutorial_space(cls):
        """Sets parameter space to
        use the tutorial space"""
        cls.empty_space()

        cls.add(
            "activation",
            ["identity", "logistic", "tanh", "relu"],
            "Activation function for the hidden layer of neural network.",
        )

        cls.add(
            "alpha",
            [10 ** i for i in range(-6, 0)],
            "L2 penalty regularization parameter",
        )

        cls.add(
            "C",
            [10 ** i for i in range(1, 7)],
            "Squared L2 penalty regularization parameter",
        )

        cls.add(
            "gamma",
            [10 ** i for i in range(-6, 7)] + ["scale", "auto"],
            "Kernel coefficient for 'rbf', 'poly', 'sigmoid'.",
        )

        cls.add(
            "kernel",
            ["linear", "poly", "rbf", "sigmoid"],
            "Kernel type to be used by SVC",
        )

        cls.add("learning_rate", [10 ** i for i in range(-6, 1)], "The learning rate")

        cls.add("max_depth", list(range(1, 31)) + [None], "Maximum depth of tree")

        cls.add(
            "max_iter",
            list(range(200, 1001, 100)),
            "Number of iterations for the algorithm",
        )

        cls.add(
            "n_estimators", list(range(50, 1001, 50)), "Number of trees in the forest"
        )

        cls.add("n_neighbors", list(range(1, 31)), "Number of neighbors to use")

        cls.add(
            "solver",
            ["lbfgs", "sgd", "adam"],
            "The solver for weight optimization of neural network.",
        )

        cls.add(
            "var_smoothing",
            [10 ** i for i in range(-18, 19)],
            "Portion of largest variance to be used for smoothing",
        )

    @classmethod
    def use_default_space(cls):
        cls.empty_space()

        # TODO: Many of these parameters' ranges were set arbitrarily.
        # The max_param were especially arbitrary as some of the documentation
        # do not give a suggested range.

        cls.add("algorithm", [], "string code for the algorithm")

        cls.add(
            "activation",
            ["identity", "logistic", "tanh", "relu"],
            "Activation function for the hidden layer of neural network.",
        )

        cls.add(
            "alpha",
            [10 ** i for i in range(-6, 0)],
            "L2 penalty regularization parameter",
        )

        cls.add(
            "C",
            [10 ** i for i in range(1, 7)],
            "Squared L2 penalty regularization parameter",
        )

        cls.add(
            "gamma",
            [10 ** i for i in range(-6, 7)] + ["scale", "auto"],
            "Kernel coefficient for 'rbf', 'poly', 'sigmoid'.",
        )

        cls.add(
            "kernel",
            ["linear", "poly", "rbf", "sigmoid"],
            "Kernel type to be used by SVC",
        )

        cls.add("learning_rate", [10 ** i for i in range(-6, 1)], "The learning rate")

        cls.add("max_depth", list(range(1, 31)) + [None], "Maximum depth of tree")

        cls.add(
            "max_iter",
            list(range(100, 1001, 100)),
            "Number of iterations for the algorithm",
        )

        cls.add(
            "n_estimators", list(range(50, 1001, 50)), "Number of trees in the forest"
        )

        cls.add("n_neighbors", list(range(1, 31)), "Number of neighbors to use")

        cls.add(
            "solver",
            ["lbfgs", "sgd", "adam"],
            "The solver for weight optimization of neural network.",
        )

        cls.add(
            "var_smoothing",
            [10 ** i for i in range(-18, 19)],
            "Portion of largest variance to be used for smoothing",
        )


ClassifierParams.use_default_space()


class Classifier(algorithm):
    """Base class for classifier classes defined below.

    Functions:
    evaluate -- Run classifier algorithm to classify dataset.

    """

    algorithmspace = dict()

    def __init__(self, paramlist=None, paramindexes=[]):
        """Generate algorithm params from parameter list."""
        super().__init__()
        self.params = ClassifierParams()
        for param_name in ClassifierParams.pkeys:
            self.params[param_name] = ClassifierParams.ranges[param_name][0]

        self.paramindexes = paramindexes
        self.set_params(paramlist)

    def evaluate(self, training_set, testing_set):
        """Instance evaluate method. Needs to be overridden by subclasses."""
        self.thisalgo = Classifier.algorithmspace[self.params["algorithm"]](self.params)
        return self.thisalgo.fit_predict(training_set, testing_set)

    def pipe(self, data):
        self.thisalgo = Classifier.algorithmspace[self.params["algorithm"]](self.params)
        data.clf = self.thisalgo.create_clf()
        return data

    @classmethod
    def add_classifier(cls, key, classifier):
        """Adds the classifier to the algorithm space."""
        ClassifierParams.ranges["algorithm"].append(key)
        cls.algorithmspace[key] = classifier

    @classmethod
    def set_search_space(cls, algorithm_space, parameter_space):    
        """
        parameter_space: a dictionary where the keys will be used
        as the parameter names and the values are tuples
        (ranges, description), where ranges is an array of all
        possible values and description is a string that describes
        the parameters purpose.
        algorithm_space: a dictionary where the keys will be
        the names of the algorithms and the value is a ClassifierContainer
        """
        # TODO: Consistency check
        # TODO: Print warnings for unused parameters

        ClassifierParams.empty_space()
        ClassifierParams.add("algorithm", [], "string code for the algorithm")

        for param_name in parameter_space:
            ranges, description = parameter_space[param_name]
            ClassifierParams.add(param_name, ranges, description)
        cls.algorithmspace = dict()
        for algo_name in algorithm_space:
            container = algorithm_space[algo_name]
            cls.add_classifier(algo_name, container)
        
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
        cls.add_classifier("Decision Tree", DecisionTreeContainer)
        cls.add_classifier("Gaussian Naive Bayes", GaussianNBContainer)
        cls.add_classifier("Gaussian Process", GaussianProcessContainer)
        cls.add_classifier("K Nearest Neighbors", KNeighborsContainer)
        cls.add_classifier("MLP Neural Network", MLPContainer)
        cls.add_classifier("Quadratic Discriminant Analysis", QDAContainer)
        cls.add_classifier("Random Forest", RandomForestContainer)
        cls.add_classifier("SVC", SVCContainer)

    @classmethod
    def use_dhahri_space(cls):
        """Sets algorithm and parameter space to
        match the search space use in Dhahri 2019"""
        # maybe.....this is fine as long as the parameter space tightens???
        # TODO: does nothing
        # TODO: might need to shrink parameterspace to match those
        # needed for the tutorial space...
        ClassifierParams.use_default_space()
        cls.algorithmspace = dict()
        cls.add_classifier("Ada Boost", AdaBoostContainer)
        cls.add_classifier("Decision Tree", DecisionTreeContainer)
        cls.add_classifier("Extra Trees", ExtraTreesContainer)
        cls.add_classifier("Gaussian Naive Bayes", GaussianNBContainer)
        cls.add_classifier("Gradient Boosting", GradientBoostingContainer)
        cls.add_classifier("Linear Discriminant Analysis", LDAContainer)
        cls.add_classifier("Logistic Regression", LogisticRegressionContainer)
        cls.add_classifier("K Nearest Neighbors", KNeighborsContainer)
        cls.add_classifier("Random Forest", RandomForestContainer)
        cls.add_classifier("SVC", SVCContainer)


class ClassifierContainer(Classifier, ABC):
    def __init__(self, clf_class, paramlist=None, *, paramindexes):
        super().__init__()
        self.set_params(paramlist)

        self.clf_class = clf_class

        # Quick hack to inject the paramindexes field into the class
        self.paramindexes = paramindexes

    def create_clf(self):
        clf = self.clf_class()
        param_dict = self.map_param_space_to_hyper_params()
        clf.set_params(**param_dict)
        return clf

    def fit_predict(self, training_set, testing_set):
        clf = self.create_clf()
        clf.fit(training_set.X, training_set.y)
        return clf.predict(testing_set.X)

    def evaluate(self, training_set, testing_set):
        return self.fit_predict(training_set, testing_set)

    @abstractmethod
    def map_param_space_to_hyper_params(self):
        pass

class AdaBoostContainer(ClassifierContainer):
    """Perform Ada Boost classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__(
            AdaBoostClassifier,
            paramlist=paramlist,
            paramindexes=["learning_rate", "n_estimators"],
        )

        self.params["algorithm"] = "Ada Boost"
        self.params["learning_rate"] = 1
        self.params["n_estimators"] = 50
        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()
        param_dict["learning_rate"] = self.params["learning_rate"]
        param_dict["n_estimators"] = self.params["n_estimators"]

        return param_dict


Classifier.add_classifier("Ada Boost", AdaBoostContainer)


class DecisionTreeContainer(ClassifierContainer):
    """Perform Decision Tree classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__(DecisionTree, paramlist, paramindexes=["max_depth"])
        self.params["algorithm"] = "Decision Tree"
        self.params["max_depth"] = None
        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()
        param_dict["max_depth"] = self.params["max_depth"]
        return param_dict


Classifier.add_classifier("Decision Tree", DecisionTreeContainer)


class ExtraTreesContainer(ClassifierContainer):
    """Perform Guassian Process classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__(
            ExtraTreesClassifier, paramlist, paramindexes=["max_depth", "n_estimators"]
        )

        self.params["algorithm"] = "Extra Trees"
        self.params["max_depth"] = None
        self.params["n_estimators"] = 100
        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()
        param_dict["max_depth"] = self.params["max_depth"]
        param_dict["n_estimators"] = self.params["n_estimators"]

        return param_dict


Classifier.add_classifier("Extra Trees", ExtraTreesContainer)


class GaussianNBContainer(ClassifierContainer):
    """Perform Gaussian Naive Bayes classification algorithm."""

    def __init__(self, paramlist=None):
        """Gaussian Naive Bayes has no parameters in the example."""

        super().__init__(GaussianNB, paramlist, paramindexes=["var_smoothing"])

        self.params["algorithm"] = "Gaussian Naive Bayes"
        self.params["var_smoothing"] = 1e-9

        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()
        param_dict["var_smoothing"] = self.params["var_smoothing"]
        return param_dict


Classifier.add_classifier("Gaussian Naive Bayes", GaussianNBContainer)


class GaussianProcessContainer(ClassifierContainer):
    """Perform Guassian Process classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__(GaussianProcessClassifier, paramlist, paramindexes=[])
        self.params["algorithm"] = "Gaussian Process"
        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()

        return param_dict


Classifier.add_classifier("Gaussian Process", GaussianProcessContainer)


class GradientBoostingContainer(ClassifierContainer):
    """Perform Gradient Boosting classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__(
            GradientBoostingClassifier,
            paramlist,
            paramindexes=["learning_rate", "n_estimators"],
        )

        self.params["algorithm"] = "Gradient Boosting"
        self.params["learning_rate"] = 0.1
        self.params["n_estimators"] = 100
        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()
        param_dict["learning_rate"] = self.params["learning_rate"]
        param_dict["n_estimators"] = self.params["n_estimators"]

        return param_dict


Classifier.add_classifier("Gradient Boosting", GradientBoostingContainer)


class KNeighborsContainer(ClassifierContainer):
    def __init__(self, paramlist=None):
        super().__init__(KNeighborsClassifier, paramlist, paramindexes=["n_neighbors"])
        self.params["algorithm"] = "K Nearest Neighbors"
        self.params["n_neighbors"] = 5
        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()
        param_dict["n_neighbors"] = self.params["n_neighbors"]
        return param_dict


Classifier.add_classifier("K Nearest Neighbors", KNeighborsContainer)


class LDAContainer(ClassifierContainer):
    """Perform LDA classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__(LinearDiscriminantAnalysis, paramlist, paramindexes=[])

        self.params["algorithm"] = "Linear Discriminant Analysis"

        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()

        return param_dict


Classifier.add_classifier("Linear Discriminant Analysis", LDAContainer)


class LogisticRegressionContainer(ClassifierContainer):
    """Perform Logistic Regression classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__(LogisticRegression, paramlist, paramindexes=["C", "max_iter"])

        self.params["algorithm"] = "Logistic Regression"
        self.params["C"] = 1
        self.params["max_iter"] = 100

        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()
        param_dict["C"] = self.params["C"]
        param_dict["max_iter"] = self.params["max_iter"]

        return param_dict


Classifier.add_classifier("Logistic Regression", LogisticRegressionContainer)


class MLPContainer(ClassifierContainer):
    """Perform MLP Neural Network classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__(
            MLPClassifier,
            paramlist,
            paramindexes=["activation", "alpha", "max_iter", "solver"],
        )
        self.params["algorithm"] = "MLP Neural Network"
        self.params["activation"] = "relu"
        self.params["alpha"] = 1
        self.params["max_iter"] = 200
        self.params["solver"] = "adam"

        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()
        param_dict["activation"] = self.params["activation"]
        param_dict["alpha"] = self.params["alpha"]
        param_dict["max_iter"] = self.params["max_iter"]
        param_dict["solver"] = self.params["solver"]

        return param_dict


Classifier.add_classifier("MLP Neural Network", MLPContainer)


class QDAContainer(ClassifierContainer):
    """Perform QDA classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__(QuadraticDiscriminantAnalysis, paramlist, paramindexes=[])

        self.params["algorithm"] = "Quadratic Discriminant Analysis"

        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()

        return param_dict


Classifier.add_classifier("Quadratic Discriminant Analysis", QDAContainer)


class RandomForestContainer(ClassifierContainer):
    """Perform Random Forest classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__(
            RandomForestClassifier,
            paramlist,
            paramindexes=["max_depth", "n_estimators"],
        )

        self.params["algorithm"] = "Random Forest"
        self.params["max_depth"] = None
        self.params["n_estimators"] = 100
        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()
        param_dict["max_depth"] = self.params["max_depth"]
        param_dict["n_estimators"] = self.params["n_estimators"]

        return param_dict


Classifier.add_classifier("Random Forest", RandomForestContainer)


class SVCContainer(ClassifierContainer):
    """Perform SVC classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__(SVC, paramlist, paramindexes=["C", "gamma", "kernel"])

        self.params["algorithm"] = "SVC"
        self.params["C"] = 1
        self.params["gamma"] = "scale"
        self.params["kernel"] = "rbf"

        self.set_params(paramlist)

    def map_param_space_to_hyper_params(self):
        param_dict = dict()
        param_dict["kernel"] = self.params["kernel"]
        param_dict["C"] = self.params["C"]
        param_dict["gamma"] = self.params["gamma"]
        if param_dict["kernel"] == "poly":
            # The poly kernel can take a very long time to run.
            # A max iteration will help force it to terminate.
            param_dict["max_iter"] = self.params["max_iter"] * 1e5

        return param_dict


Classifier.add_classifier("SVC", SVCContainer)

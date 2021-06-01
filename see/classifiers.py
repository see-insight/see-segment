"""Classifer algorithm library designed to classify images with a searchable parameter space.
 This libary actually does not incode the search code itself, instead it just defines
  the search parameters and the evaluation funtions."""

import numpy as np
#import sklearn

from sklearn.tree import DecisionTreeClassifier as DecisionTree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as kNearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from see.base_classes import param_space, algorithm


class ClassifierParams(param_space):
    """Parameter space for classifiers
    """
    descriptions = dict()
    ranges = dict()


ClassifierParams.add('algorithm',
                     [],
                     "string code for the algorithm")

ClassifierParams.add('h',
                     [float(i) / 256 for i in range(0, 256)],
                     "step size in the mesh"
                     )


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
        self.set_params(paramlist)

    # Input: labelled dataset
    # Output: Tuple with prediction probabilities, score
    def evaluate(self, dataset):
        """Instance evaluate method. Needs to be overridden by subclasses."""
        # Default method does nothing and returns score 0 with not prediction
        return [dataset, 0]

    @classmethod
    def add_classifier(cls, key, classifier):
        """Adds the classifier to the algorithm space."""
        ClassifierParams.ranges['algorithm'].append(key)
        cls.algorithmspace[key] = classifier


class GaussianNBClassifier(Classifier):
    """Perform Gaussian Naive Bayes classification algorithm."""

    def __init__(self, paramlist=None):
        """Gaussian Naive Bayes has no parameters in the example."""
        super().__init__()
        self.params = ClassifierParams()

        self.params["h"] = .02
        self.set_params(paramlist)

    def evaluate(self, dataset):
        """The evaluate function for Gaussian Naive Bayes"""

        h = self.params["h"]  # step size in the mesh

        clf = GaussianNB()
        X, y = dataset
        X = StandardScaler().fit_transform(X)

        # TODO: test_size and random_state are hard-coded
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.4, random_state=42)

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # TODO: Generating the mesh grid can be a Class method
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        return [Z, score]


Classifier.add_classifier('Gaussian Naive Bayes', GaussianNBClassifier)


class KNeighborsClassifier(Classifier):
    """Perform K Nearest-Neighbors classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__()
        self.params = ClassifierParams()

        self.params["h"] = .02
        self.params["k"] = 3
        self.set_params(paramlist)

    def evaluate(self, dataset):
        """The evaluate function for K Nearest-Neighbors"""

        h = self.params["h"]  # step size in the mesh

        clf = kNearestNeighbors(self.params["k"])
        X, y = dataset
        X = StandardScaler().fit_transform(X)

        # TODO: test_size and random_state are hard-coded
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.4, random_state=42)

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # TODO: Generating the mesh grid can be a Class method
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        return [Z, score]


Classifier.add_classifier('K Nearest Neighbors', KNeighborsClassifier)


class DecisionTreeClassifier(Classifier):
    """Perform Decision Tree classification algorithm."""

    def __init__(self, paramlist=None):
        super().__init__()
        self.params = ClassifierParams()

        self.params["h"] = .02
        self.params["max_depth"] = 5
        self.set_params(paramlist)

    def evaluate(self, dataset):
        """The evaluate function for K Nearest-Neighbors"""

        h = self.params["h"]  # step size in the mesh

        clf = DecisionTree(max_depth=self.params["max_depth"])
        X, y = dataset
        X = StandardScaler().fit_transform(X)

        # TODO: test_size and random_state are hard-coded
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.4, random_state=42)

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # TODO: Generating the mesh grid can be a Class method
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        return [Z, score]


Classifier.add_classifier('Decision Tree', DecisionTreeClassifier)

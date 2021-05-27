"""Classifer algorithm library designed to classify images with a searchable parameter space.
 This libary actually does not incode the search code itself, instead it just defines
  the search parameters and the evaluation funtions."""

import numpy as np
#import sklearn

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from see.base_classes import param_space, algorithm


class ClassifierParams(param_space):
    """Parameter space for classifiers
    """
    descriptions = dict()
    ranges = dict()


ClassifierParams.add('algorithm',
                     [],
                     "string code for the algorithm")


class Classifier(algorithm):
    """Base class for classifier classes defined below.

    Functions:
    evaluate -- Run classifier algorithm to...

    """

    algorithmspace = dict()

    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        super().__init__()
        self.params = ClassifierParams()
        self.set_params(paramlist)

    def evaluate(self, dataset):
        """Instance evaluate method. Needs to be overridden by subclasses."""
        print("Default do nothing evaluate method")
        print(self.params)
        print(dataset)

    @classmethod
    def add_classifier(cls, key, classifier):
        """Adds the classifier to the algorithm space."""
        ClassifierParams.ranges['algorithm'].append(key)
        cls.algorithmspace[key] = classifier


class NaiveBayes(Classifier):
    """Perform Gaussian Naive Bayes classification algorithm."""

    def __init__(self, paramlist=None):
        """Gaussian Naive Bayes has no parameters in the example."""
        super().__init__()
        self.set_params(paramlist)

    def evaluate(self, dataset):
        """The evaluate function for Gaussian Naive Bayes"""

        h = .02  # step size in the mesh

        clf = GaussianNB()
        X, y = dataset
        X = StandardScaler().fit_transform(X)

        clf.fit(X, y)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        return Z


Classifier.add_classifier('Naive Bayes', NaiveBayes)

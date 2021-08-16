"""This module provides utilities to using the see classification pipeline.
"""

from sklearn.model_selection import train_test_split
from sklearn.base import is_classifier

from see.base_classes import pipedata


class ClassifyDataset(pipedata):
    """
    The dataset object that should be used in the classifier pipeline.

    Attributes
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to fit or predict on.

    y : array-like of shape (n_samples,)
        The target label to predict.

    Notes
    -----
    Data (X) should be preprocessed before building this object.
    """

    def __init__(self, X, y):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples,)
        """
        self.X = X
        self.y = y


class PipelineClassifyDataset(pipedata):
    """
    The dataset object that should be used in the classifier pipeline.

    Attributes
    ----------
    training_set : ClassifyDataset
        The dataset to train on.

    testing_set : ClassifyDataset, default=None
        The dataset to test on.

    clf : Scikit-learn Classifier object or None, default=None
        The classifier object that will be used to on
        the training and testing sets.

    fitness : float or None, default=None
        The fitness score of the classifier.
    """

    def __init__(self, training_set, testing_set=None, clf=None, fitness=None):
        """
        Parameters
        ----------
        training_set : ClassifyDataset
            The dataset to train on.

        testing_set : ClassifyDataset, default=None
            The dataset to test on.
        """
        self.training_set = training_set
        self.testing_set = testing_set
        self._clf = clf
        self._fitness = fitness

    @property
    def clf(self):
        return self._clf

    @clf.setter
    def clf(self, clf):
        if self._clf is not None:
            print("WARNING: Attached classifier has changed multiple times during pipeline")
        if not is_classifier(clf):
            raise ValueError("clf must be a Scikit-learn Classifier object")
        self._clf = clf

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        if self._fitness is not None:
            print("WARNING: fitness has changed multiple times during pipeline")
        if not isinstance(fitness, int) and not isinstance(fitness, float):
            raise ValueError("Fitness must be int or float.")
        self._fitness = fitness


def generate_pipeline_dataset(X, y):
    """
    Create dataset object that should be used in the classifier pipeline.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The data to fit or predict on.

    y : array-like of shape (n_samples,)
        The target label to predict.

    Returns
    -------
    dataset : PipelineClassifyDataset
        Dataset object to be used/passed into the genetic search pipeline
        for classifiers.
        X, y are used to build dataset.training_set.
        dataset.testing_set is None.

    Notes
    -----
    Data (X) should be preprocessed before applying this function.
    """

    dataset = PipelineClassifyDataset(ClassifyDataset(X, y))
    return dataset


def generate_train_test_set(X, y, random_state=42, test_size=0.4):
    """
    Split data into training and testing sets

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The data to fit or predict on.

    y : array-like of shape (n_samples,)
        The target label to predict.

    random_state : int or None, default=42
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    test_size : float, default=0.4
        Represents the portion of the dataset to include in the testing set.

    Returns
    -------
    dataset : PipelineClassifyDataset

    Notes
    -----
    This function is essentially a wrapper of the
    `sklearn.model_selection.train_test_split` function found at
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.

    Data (X) should be preprocessed before applying this function.
    """

    # Split data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    training_set = ClassifyDataset(X_train, y_train)
    testing_set = ClassifyDataset(X_test, y_test)

    dataset = PipelineClassifyDataset(training_set, testing_set)

    return dataset

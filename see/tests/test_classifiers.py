"""This runs unit tests for functions that can be found in classifiers.py."""

import numpy as np

from sklearn.datasets import make_moons  # , make_circles, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as DecisionTree

from sklearn.model_selection import train_test_split
from see.base_classes import pipedata

from see import classifiers


def test_gaussian_naive_bayes():
    """Unit test for Gaussian Naive Bayes classifer algorithm.
    Check if evaluate function output is same as it would be
    running the sklearn function manually."""

    # Generate dataset
    dataset = make_moons(noise=0.3, random_state=0)
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=42)

    clf = GaussianNB()

    clf.fit(X_train, y_train)  # Train classifier

    # manual sklearn categorizations
    expected_predictions = clf.predict(X_test)

    # see-classify categorizations
    see_clf = classifiers.GaussianNBClassifier()
    testing_set = pipedata()
    testing_set.X = X_test
    testing_set.y = y_test

    training_set = pipedata()
    training_set.X = X_train
    training_set.y = y_train
    
    actual_predictions = see_clf.evaluate(training_set, testing_set)

    assert len(actual_predictions) == len(expected_predictions)
    assert (actual_predictions == expected_predictions).all()

def test_nearest_neighbor():
    """Unit test for Nearest Neighbors classifier algorithm.
    Check if evaluate function output is same as it would be
    running the sklearn function manually."""

    # Generate dataset
    dataset = make_moons(noise=0.3, random_state=0)
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=42)

    n_neighbors = 3
    clf = KNeighborsClassifier(n_neighbors)

    
    clf.fit(X_train, y_train)  # Train classifier

    # manual sklearn categorizations
    expected_predictions = clf.predict(X_test)

    # see-classify categorizations
    testing_set = pipedata()
    testing_set.X = X_test
    testing_set.y = y_test

    training_set = pipedata()
    training_set.X = X_train
    training_set.y = y_train
        
    see_clf = classifiers.KNeighborsClassifier()
    actual_predictions = see_clf.evaluate(training_set, testing_set)

    assert see_clf.params["n_neighbors"] == n_neighbors
    assert len(actual_predictions) == len(expected_predictions)
    assert (actual_predictions == expected_predictions).all()

def test_decision_tree():
    """Unit test for Decision Tree classifier algorithm.
    Check if evaluate function output is same as it would be
    running the sklearn function manually."""

    # Generate dataset
    dataset = make_moons(noise=0.3, random_state=0)
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=42)

    clf = DecisionTree(max_depth=5)

    clf.fit(X_train, y_train)  # Train classifier

    # manual sklearn categorizations
    expected_predictions = clf.predict(X_test)

    # see-classify categorizations
    testing_set = pipedata()
    testing_set.X = X_test
    testing_set.y = y_test

    training_set = pipedata()
    training_set.X = X_train
    training_set.y = y_train

    see_clf = classifiers.DecisionTreeClassifier()
    actual_predictions = see_clf.evaluate(training_set, testing_set)

    assert len(actual_predictions) == len(expected_predictions)
    assert (actual_predictions == expected_predictions).all()




def test_random_forest():
    """Unit test for Decision Tree classifier algorithm.
    Check if evaluate function output is same as it would be
    running the sklearn function manually."""

    # TODO: This test sometimes fails. I think it is because the algorithm
    # has a randomness to it. Look into the random_state parameter
    # to control this.
    
    # Generate dataset
    dataset = make_moons(noise=0.3, random_state=0)
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=42)


    clf = RandomForestClassifier(max_depth=5, n_estimators=10)

    clf.fit(X_train, y_train)  # Train classifier

    # manual sklearn categorizations
    expected_predictions = clf.predict(X_test)

    # manual sklearn fitness score
    testing_set = pipedata()
    testing_set.X = X_test
    testing_set.y = y_test

    training_set = pipedata()
    training_set.X = X_train
    training_set.y = y_train

    # see-classify categorizations
    see_clf = classifiers.RandomForestContainer()
    actual_predictions = see_clf.evaluate(training_set, testing_set)

    assert len(actual_predictions) == len(expected_predictions)
    assert (actual_predictions == expected_predictions).all()


def test_mlp():
    """Unit test for Decision Tree classifier algorithm.
    Check if evaluate function output is same as it would be
    running the sklearn function manually."""

    # TODO: This test sometimes fails. I think it is because the algorithm
    # has a randomness to it. Look into the random_state parameter
    # to control this.
    
    # Generate dataset
    h = 0.02

    dataset = make_moons(noise=0.3, random_state=0)
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=42)

    clf = MLPClassifier(alpha=1, max_iter=1000)

    clf.fit(X_train, y_train)  # Train classifier

    # manual sklearn categorizations
    expected_predictions = clf.predict(X_test)
   
    # manual sklearn fitness score
    testing_set = pipedata()
    testing_set.X = X_test
    testing_set.y = y_test

    training_set = pipedata()
    training_set.X = X_train
    training_set.y = y_train
   
    # see-classify categorizations
    see_clf = classifiers.MLPContainer()
    actual_predictions = see_clf.evaluate(training_set, testing_set)

    assert len(actual_predictions) == len(expected_predictions)
    assert (actual_predictions == expected_predictions).all()

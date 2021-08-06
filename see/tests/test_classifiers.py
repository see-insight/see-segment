"""This runs unit tests for functions that can be found in classifiers.py."""

import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as DecisionTree

from see.classifier_helpers.fetch_data import generate_tutorial_data
from see.classifier_helpers.helpers import generate_pipeline_dataset

from see.base_classes import pipedata

from see import classifiers


def test_gaussian_naive_bayes_defaults():
    """Unit test for Gaussian Naive Bayes classifer algorithm.
    Check if classifier container with default parameters 
    performs the same as running the corresponding sklearn algorithm
    with their default parameters."""

    # Generate dataset
    datasets = generate_tutorial_data()

    clf = GaussianNB()

    # manual sklearn categorizations
    expected_predictions = []
    for ds_name in datasets:
        X, y = datasets[ds_name]
        clf.fit(X, y)  # Train classifier
        expected_predictions.append(clf.predict(X))

    clf_container = classifiers.GaussianNBContainer()
    
    # Check that default params are equal
    assert (clf_container.create_clf().get_params() == clf.get_params())
    
    # Check that the evaluate function works correctly
    for i, ds_name in enumerate(datasets):
        X, y = datasets[ds_name]
        pipeline_ds = generate_pipeline_dataset(X, y)
        actual_predictions = clf_container.evaluate(pipeline_ds.training_set, pipeline_ds.testing_set)
        assert len(actual_predictions) == len(expected_predictions[i])
        assert (actual_predictions == expected_predictions[i]).all()

def test_nearest_neighbor_defaults():
    """Unit test for Nearest Neighbors classifer algorithm.
    Check if classifier container with default parameters 
    performs the same as running the corresponding sklearn algorithm
    with their default parameters."""

    # Generate dataset
    datasets = generate_tutorial_data()

    clf = KNeighborsClassifier()

    # manual sklearn categorizations
    expected_predictions = []
    for ds_name in datasets:
        X, y = datasets[ds_name]
        clf.fit(X, y)  # Train classifier
        expected_predictions.append(clf.predict(X))

    clf_container = classifiers.KNeighborsContainer()
    
    # Check that default params are equal
    assert (clf_container.create_clf().get_params() == clf.get_params())
    
    # Check that the evaluate function works correctly
    for i, ds_name in enumerate(datasets):
        X, y = datasets[ds_name]
        pipeline_ds = generate_pipeline_dataset(X, y)
        actual_predictions = clf_container.evaluate(pipeline_ds.training_set, pipeline_ds.testing_set)
        assert len(actual_predictions) == len(expected_predictions[i])
        assert (actual_predictions == expected_predictions[i]).all()

def test_decision_tree_defaults():
    """Unit test for Decision Tree classifier algorithm.
    Check if classifier container with default parameters 
    performs the same as running the corresponding sklearn algorithm
    with their default parameters."""
    
    # Generate dataset
    datasets = generate_tutorial_data()

    random_state = 21
    clf = DecisionTree(random_state=random_state)

    # manual sklearn categorizations
    expected_predictions = []
    for ds_name in datasets:
        X, y = datasets[ds_name]
        clf.fit(X, y)  # Train classifier
        expected_predictions.append(clf.predict(X))

    clf_container = classifiers.DecisionTreeContainer(random_state=random_state)
    
    # Check that default params are equal
    assert (clf_container.create_clf().get_params() == clf.get_params())
    
    # Check that the evaluate function works correctly
    for i, ds_name in enumerate(datasets):
        X, y = datasets[ds_name]
        pipeline_ds = generate_pipeline_dataset(X, y)
        actual_predictions = clf_container.evaluate(pipeline_ds.training_set, pipeline_ds.testing_set)
        assert len(actual_predictions) == len(expected_predictions[i])
        assert (actual_predictions == expected_predictions[i]).all()
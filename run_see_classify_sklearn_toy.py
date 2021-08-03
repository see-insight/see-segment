"""
The purpose of this script is to run collect data on the performance of
SEE-Classify applied to the Sklearn tutorial on Classification Algorithms
linked here: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html.

<trial-number>,<generation-number>,<best-hof-fitness>

The default parameters for the Genetic Algorithm are
Dataset (--dataset-name) = 'moons'
Population Size (--pop-size) = 10
Number of Generations (--num-gen) = 10
Number of Trials (--num-trials) = 100
"""

import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from see import GeneticSearch
from see.classifiers import Classifier
from see.classifier_fitness import ClassifierFitness
from see.classifier_helpers import helpers
from see.classifier_helpers.fetch_data import generate_tutorial_data
from see.Workflow import workflow

possible_dataset_names = ["moons", "circles", "linearly_separable"]

parser = argparse.ArgumentParser(description="Create some csv data files.")

parser.add_argument(
    "--dataset-name",
    default="moons",
    choices=possible_dataset_names,
    help="Name of the dataset to generate data for (default: moons)",
)

parser.add_argument(
    "--num-gen",
    default=10,
    type=int,
    help="number of generations to run genetic search (default: 20)",
)

parser.add_argument(
    "--pop-size",
    default=10,
    type=int,
    help="population size of each generation to run genetic search (default: 20)",
)

parser.add_argument(
    "--num-trials",
    default=100,
    type=int,
    help="number of trials to run genetic search (default: 100)",
)

args = parser.parse_args()

# Create Data: Sklearn tutorial toy datasets
ds_name = args.dataset_name

datasets = generate_tutorial_data()
X, y = datasets[ds_name]

# Split data into training and testing sets and
# create a dataset object that can be fed into the pipeline
pipeline_dataset = helpers.generate_train_test_set(X, y)

NUM_GENERATIONS = args.num_gen
NUM_TRIALS = args.num_trials
POP_SIZE = args.pop_size

print("Running {} Dataset".format(ds_name))
print("GA running for {} generations with population size of {}".format(NUM_GENERATIONS, POP_SIZE))
print("Size of dataset: {}".format(len(X)))
print("Size of training set: {}".format(len(pipeline_dataset.training_set.y)))
print("Size of testing set: {}".format(len(pipeline_dataset.testing_set.y)))
print("\n")

# Initialize Algorithm Space and Workflow
Classifier.use_tutorial_space()

# Check algorithm space
algorithm_space = Classifier.algorithmspace
print("Algorithm Space: ")
print(list(algorithm_space.keys()))
print("\n")

workflow.addalgos([Classifier, ClassifierFitness])
wf = workflow()

for i in range(NUM_TRIALS):
    print("Running trial number {}".format(i))
    my_evolver = GeneticSearch.Evolver(workflow, pipeline_dataset, pop_size=POP_SIZE)
    my_evolver.run(
        ngen=NUM_GENERATIONS,
	print_raw_data=True
    )

"""
The purpose of this script is to run collect data on the performance of
SEE-Classify applied to the Sklearn tutorial on Classification Algorithms
linked here: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html.
This script will generate a CSV file, where each line contains the following information:

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
from sklearn.datasets import make_moons, make_circles, make_classification
from see import GeneticSearch
from see.base_classes import pipedata
from see.classifiers import Classifier
from see.classifier_fitness import ClassifierFitness
from see.classifier_helpers import helpers
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
    "--filename-tail",
    default="0",
    help="tail to add to the end of the filenames generated (default: <filename>_0.csv)",
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

parser.add_argument(
    "--num-samples",
    default=100,
    type=int,
    help="number of samples to generate in datasets (default: 100)",
)

args = parser.parse_args()

## PRINT MESSAGE

## WAIT MESSAGE...

# Initialize Algorithm Space and Workflow
Classifier.use_tutorial_space()

algorithm_space = Classifier.algorithmspace
print(algorithm_space)  # Check algorithm space

workflow.addalgos([Classifier, ClassifierFitness])
wf = workflow()


# Create Data: Sklearn tutorial toy datasets
X = None
y = None

ds_name = args.dataset_name
n_samples = args.num_samples

if ds_name == "moons":
    X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=0)
elif ds_name == "circles":
    X, y = make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1)
elif ds_name == "linearly_separable":
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
else:
    # We do not expect to reach this case. arg_parser should protect us from ever reaching
    # this case. However, for the sake of consistency, we add the default case.
    print("WARNING: Unexpected dataset name {}".format(ds_name))
    raise ValueError("Dataset name must be one of {}".format(possible_dataset_names))

# Preprocess data
X = StandardScaler().fit_transform(X)

# Split data into training and testing sets and
# create a dataset object that can be fed into the pipeline
temp = helpers.generate_train_test_set(X, y)
testing_set = temp.testing_set
pipeline_dataset = helpers.generate_train_test_set(
    temp.training_set.X, temp.training_set.y, test_size=0.25
)

NUM_GENERATIONS = args.num_gen
NUM_TRIALS = args.num_trials
POP_SIZE = args.pop_size

print("Running {} Dataset".format(ds_name))
# best_algo_fitness = []
for i in range(NUM_TRIALS):
    print("Running trial number {}".format(i))
    my_evolver = GeneticSearch.Evolver(workflow, pipeline_dataset, pop_size=POP_SIZE)
    my_evolver.run(
        ngen=NUM_GENERATIONS,
        line_template="{trial_num},{},{}\n".format("{}", "{}", trial_num=i),
        output_best_hof_fitness_to_file=True,
        output_filename="{}_fitness_{}.csv".format(ds_name, args.filename_tail),
    )
    # best_algo_fitness.append([my_evolver.hof[0].fitness.values[0], my_evolver.hof[0]])
    # Evaluate performance of hall of fame over testing set.
    for j, ind in enumerate(my_evolver.hof):
        algo_name = ind[0]
        param_list = ind

        clf = Classifier.algorithmspace[algo_name](param_list)

        # Reform training set
        training_set = pipedata()
        training_set.X = np.concatenate((pipeline_dataset.training_set.X, pipeline_dataset.testing_set.X), axis=0)
        training_set.y = np.concatenate((pipeline_dataset.training_set.y, pipeline_dataset.testing_set.y), axis=0)
        print(len(testing_set.X))
        predictions = clf.evaluate(training_set, testing_set)

        score = ClassifierFitness().evaluate(predictions, testing_set.y)
        print(
            "# Evaluation TEST for trial={},individual={},score={}".format(i, j, score)
        )

# best_algo_fitness = np.array(best_algo_fitness)
# print(best_algo_fitness)
# print(best_algo_fitness[:,0])
# print("Average fitness: {} ".format(sum(best_algo_fitness[:,0])/NUM_TRIALS))
# print("Min fitness: {} ".format(min(best_algo_fitness[:,0])))
# print("Max fitness: {} ".format(max(best_algo_fitness[:,0])))

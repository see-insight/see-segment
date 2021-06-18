"""
The purpose of this script is to run experiments on each of the datasets...
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

parser = argparse.ArgumentParser(description='Create some csv data files.')

parser.add_argument('--filename-tail', default='0',
                    help='tail to add to the end of the filenames generated (default: <filename>_0.csv)')

parser.add_argument('--num-gen', default=20, type=int,
        help='number of generations to run genetic search (default: 20)')

parser.add_argument('--pop-size', default=20, type=int,
        help='population size of each generation to run genetic search (default: 20)')

args = parser.parse_args()

# Initialize Algorithm Space and Workflow
algorithm_space = Classifier.algorithmspace

workflow.addalgos([Classifier, ClassifierFitness])
wf = workflow()


# Create Data: Sklearn tutorial toy datasets
# Moons
moons_ds = pipedata()
moons_ds.name = 'Moons Dataset'
moons_ds.X, moons_ds.y = make_moons(noise=0.3, random_state=0)

# Circles
circles_ds = pipedata()
circles_ds.name = 'Circles Dataset'
circles_ds.X, circles_ds.y = make_circles(
    noise=0.2, factor=0.5, random_state=1)

# Linearly Seperable dataset
lin_ds = pipedata()
lin_ds.X, lin_ds.y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                         random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
lin_ds.name = 'Linearly Separable Dataset'
lin_ds.X += 2 * rng.uniform(size=lin_ds.X.shape)

datasets = [moons_ds, circles_ds, lin_ds]
validation_sets = []

# Preprocess data
for ds in datasets:
    ds.X = StandardScaler().fit_transform(ds.X)

# Split datasets into training, testing, and validation sets
for i, ds in enumerate(datasets):
    temp = helpers.generate_train_test_set(ds.X, ds.y)
    validation_sets.append(temp.testing_set)
    datasets[i] = helpers.generate_train_test_set(
        temp.training_set.X, temp.training_set.y)
    datasets[i].name = ds.name

NUM_GENERATIONS = args.num_gen
POP_SIZE = args.pop_size
hof_per_dataset = []

for ds in datasets:
    print('Running ', ds.name)
    my_evolver = GeneticSearch.Evolver(workflow, ds, pop_size=POP_SIZE)
    my_evolver.run(ngen=NUM_GENERATIONS, print_fitness_to_file=True, print_fitness_filename="{}_fitness_{}.csv".format(ds.name, args.filename_tail))
    # Store the best solution found for each dataset
    hof_per_dataset.append(my_evolver.hof)

# Path hack so that we can import see library.
import sys, os

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np


def extract_hof_population(
    population_filepath,
    hof_filepath,
    num_gen=100,
    pop_size=100,
    num_trials=30,
    hof_size=10,
):
    population_df = pd.read_csv(population_filepath, header=None, delimiter=";")
    hof_df = pd.read_csv(hof_filepath, header=None, delimiter=";")
    return population_df, hof_df


def extract_hof_population(
    hof_filepath,
    num_gen=100,
    pop_size=100,
    num_trials=30,
    hof_size=10,
):
    import pandas as pd

    hof_df = pd.read_csv(hof_filepath, header=None, delimiter=";")
    return hof_df


def create_hof_at_gen(hof_filepath, num_gen, pop_size, num_trials, hof_size, at_gen=0):
    import ast

    hof_df = extract_hof_population(
        hof_filepath, num_gen, pop_size, num_trials, hof_size
    )

    hof_list = hof_df[hof_df[0] == at_gen].to_numpy()
    hof_containers = list(map(ast.literal_eval, hof_list[:, 3]))
    return hof_containers, hof_list

def create_learning_curve(training_set, validation_set, containers):
    # learning curve
    from see.classifiers import Classifier
    import numpy as np

    train_sizes = np.linspace(30, len(training_set.X), num=10, dtype=int)
    validation_matrix = np.zeros((len(containers), len(train_sizes)))
    train_matrix = np.zeros((len(containers), len(train_sizes)))

    for index, individual in enumerate(containers):

        temp = Classifier.algorithmspace[individual[0]](individual)
        clf = temp.create_clf()
        test_scores = np.zeros(len(train_sizes))
        train_scores = np.zeros(len(train_sizes))

        for i, portion in enumerate(train_sizes):
            temp_X = training_set.X[0:portion]
            temp_y = training_set.y[0:portion]
            clf.fit(temp_X, temp_y)

            train_score = clf.score(temp_X, temp_y)
            train_scores[i] = 1 - train_score

            validation_score = clf.score(validation_set.X, validation_set.y)
            test_scores[i] = 1 - validation_score
        validation_matrix[index, :] = test_scores
        train_matrix[index, :] = train_scores
        print("Done with individual: ", index)
    return train_sizes, train_matrix, validation_matrix

def plot_learning_curve(train_sizes, train_matrix, validation_matrix, axs=None):
    import matplotlib.pyplot as plt

    if axs == None:
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    train_scores_mean = np.mean(train_matrix, axis=0)
    train_scores_std = np.std(train_matrix, axis=0)
    test_scores_mean = np.mean(validation_matrix, axis=0)
    test_scores_std = np.std(validation_matrix, axis=0)

    # Plot learning curve
    axs.grid()

    alpha = 0.3
    train_low_bound = train_scores_mean - 2 * train_scores_std
    train_low_bound[train_low_bound < 0] = 0  # Remove negative values

    blue = "#377eb8"
    red = "#e41a1c"
    gray = "#999999"

    axs.fill_between(
        train_sizes,
        train_low_bound,
        train_scores_mean + 2 * train_scores_std,
        alpha=alpha,
        color=blue,
    )

    axs.plot(
        train_sizes, train_scores_mean, "o-", color=blue, label="Mean Training Fitness"
    )

    axs.fill_between(
        train_sizes,
        test_scores_mean - 2 * test_scores_std,
        test_scores_mean + 2 * test_scores_std,
        alpha=alpha,
        color=red,
    )

    axs.plot(
        train_sizes, test_scores_mean, "x-", color=red, label="Mean Validation Fitness"
    )

    plt.xlim(0, 350)

    axs.legend(loc="best")

    axs.tick_params(which="both", direction="in", labelsize=14)

    plt.title("Evaluate Best Found Solutions", fontdict={"fontsize": 24})

    axs.set_xlabel("Training Sizes", fontdict={"fontsize": 20})
    axs.set_ylabel("Fitness Scores", fontdict={"fontsize": 20})

    fig.text(0, -0.03, "* Shaded regions are 2 STD\n from means")
    plt.tight_layout()

    return axs

def extract_stats(population_df, hof_df, num_gen, pop_size, num_trials, hof_size):
    # slice_gen allows us to plot the range between 0 and a specific generation number
    slice_gen = num_gen  # disable slice_gen and set it equal to num_gen. Use matplotlib xlim instead!

    # extracts:
    # means_of_means
    # std_sample_samples
    # hof_means_of_means
    # hof_std_sample_means
    # hof_means_of_mins
    # hof_std_of_mins

    generations = list(range(0, slice_gen + 1))

    # trials x generation number
    trial_means = np.zeros((num_trials, slice_gen + 1))

    for i in range(0, num_trials):
        sample = population_df[
            (num_gen + 1) * (pop_size) * i : (num_gen + 1) * (pop_size) * (i + 1)
        ]
        for j in range(0, slice_gen + 1):
            # Get all rows for generation j
            rows = sample[sample[0] == j]
            trial_means[i, j] = rows[2].mean()

    means_of_means = np.zeros(slice_gen + 1)
    for i in range(slice_gen + 1):
        means_of_means[i] = trial_means[:, i].mean()

    std_sample_means = np.zeros(slice_gen + 1)
    for i in range(slice_gen + 1):
        std_sample_means[i] = trial_means[:, i].std()

    # Trials x generation number
    hof_trial_means = np.zeros((num_trials, slice_gen + 1))

    for i in range(0, num_trials):
        sample = hof_df[
            (num_gen + 1) * (hof_size) * i : (num_gen + 1) * (hof_size) * (i + 1)
        ]
        for j in range(0, slice_gen + 1):
            # Get all rows for generation j
            rows = sample[sample[0] == j]
            hof_trial_means[i, j] = rows[2].mean()

    hof_means_of_means = np.zeros(slice_gen + 1)
    for i in range(slice_gen + 1):
        hof_means_of_means[i] = hof_trial_means[:, i].mean()

    hof_std_sample_means = np.zeros(slice_gen + 1)
    for i in range(slice_gen + 1):
        hof_std_sample_means[i] = hof_trial_means[:, i].std()

    # Trials x generation number
    hof_trial_mins = np.zeros((num_trials, slice_gen + 1))

    for i in range(0, num_trials):
        sample = population_df[
            (num_gen + 1) * (pop_size) * i : (num_gen + 1) * (pop_size) * (i + 1)
        ]
        for j in range(0, slice_gen + 1):
            # Get all rows for generation j
            rows = sample[sample[0] == j]
            hof_trial_mins[i, j] = rows[2].min()

    hof_means_of_mins = np.zeros(slice_gen + 1)
    for i in range(slice_gen + 1):
        hof_means_of_mins[i] = hof_trial_mins[:, i].mean()

    hof_std_of_mins = np.zeros(slice_gen + 1)
    for i in range(slice_gen + 1):
        hof_std_of_mins[i] = hof_trial_mins[:, i].std()

    return (
        means_of_means,
        std_sample_means,
        hof_means_of_means,
        hof_std_sample_means,
        hof_means_of_mins,
        hof_std_of_mins,
    )


def benchmark_generation_fitness(
    population_filepath,
    hof_filepath,
    num_gen,
    pop_size,
    num_trials,
    hof_size,
):

    population_df, hof_df = extract_hof_population(
        population_filepath, hof_filepath, num_gen, pop_size, num_trials, hof_size
    )

    (
        means_of_means,
        std_sample_means,
        hof_means_of_means,
        hof_std_sample_means,
        hof_means_of_mins,
        hof_std_of_mins,
    ) = extract_stats(population_df, hof_df, num_gen, pop_size, num_trials, hof_size)
    
    return (
        means_of_means,
        std_sample_means,
        hof_means_of_means,
        hof_std_sample_means,
        hof_means_of_mins,
        hof_std_of_mins,
    )

def plot_generation_fitness(
    population_filepath,
    hof_filepath,
    num_gen,
    pop_size,
    num_trials,
    hof_size,
    axs=None
):
    # Extract data
    (
        means_of_means,
        std_sample_means,
        hof_means_of_means,
        hof_std_sample_means,
        hof_means_of_mins,
        hof_std_of_mins,
    ) = benchmark_generation_fitness(
        population_filepath,
        hof_filepath,
        num_gen,
        pop_size,
        num_trials,
        hof_size,
    )
    # Plot data
    import matplotlib.pyplot as plt

    if axs == None:

        fig, axs = plt.subplots(1, sharex=True, figsize=(10, 5))
    
    generations = list(range(0, num_gen + 1))

    alpha = 0.3

    blue = "#377eb8"
    red = "#e41a1c"
    gray = "#999999"

    axs.plot(generations, means_of_means, "x-", color=red, label="Average Population Means")
    axs.fill_between(
        generations,
        means_of_means - 2 * std_sample_means,
        means_of_means + 2 * std_sample_means,
        alpha=alpha,
        color=red,
    )

    axs.plot(
        generations,
        hof_means_of_means,
        "o-",
        color=blue,
        label="Average mean of top 10 Best So Far",
    )
    axs.fill_between(
        generations,
        hof_means_of_means - 2 * hof_std_sample_means,
        hof_means_of_means + 2 * hof_std_sample_means,
        alpha=alpha,
        color=blue,
    )

    axs.plot(generations, hof_means_of_mins, "*-", color=gray, label="Average Best So Far")
    axs.fill_between(
        generations,
        hof_means_of_mins - 2 * hof_std_of_mins,
        hof_means_of_mins + 2 * hof_std_of_mins,
        alpha=alpha,
        color=gray,
    )

    axs.legend(loc="best")

    plt.xlim(0, num_gen)

    axs.set_title("Breast Cancer Dataset", fontdict={"fontsize": 24})
    axs.set_xlabel("Iteration Number", fontdict={"fontsize": 20})
    axs.set_ylabel("Fitness Value", fontdict={"fontsize": 20})
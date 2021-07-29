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

    axs.set_xticks(list(range(21)))
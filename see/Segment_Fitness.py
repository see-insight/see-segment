"""File Segment_Fitness.py."""

import sys
from skimage import color
import numpy as np

from see.base_classes import algorithm


def countMatches(inferred, ground_truth):
    """Map the segments in the inferred segmentation mask to the ground truth segmentation.

     mask, and record the number of pixels in each of these mappings as well as the number
     of segments in both masks.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    ground_truth -- Ground truth segmentation mask for training image.

    Outputs:
    setcounts -- Dictionary of dictionaries containing the number of pixels in
        each segment mapping.
    len(m) -- Number of segments in inferred segmentation mask.
    len(n) -- Number of segments in ground truth segmentation mask.

    """
    #print(f" {inferred.shape=} {ground_truth.shape=}")
    assert inferred.shape == ground_truth.shape
    m = set()
    n = set()
    setcounts = dict()
    for r in range(inferred.shape[0]):
        for c in range(inferred.shape[1]):
            i_key = inferred[r, c]
            m.add(i_key)
            g_key = ground_truth[r, c]
            n.add(g_key)
            if i_key in setcounts:
                if g_key in setcounts[i_key]:
                    setcounts[i_key][g_key] += 1
                else:
                    setcounts[i_key][g_key] = 1
            else:
                setcounts[i_key] = dict()
                setcounts[i_key][g_key] = 1
    return setcounts, len(m), len(n)


def countsets(setcounts):
    """For each inferred set, find the ground truth set it maps the most pixels to.

    So we start from the inferred image, and map towards the ground truth image.
    For each i_key, the g_key that it maps the most pixels to is considered True.
    In order to see what ground truth sets have a corresponding set(s) in the
    inferred
    image, we record these "true" g_keys. This number of true g_keys is the value for
    L in our fitness function.

    Keyword arguments:
    setcounts -- Dictionary of dictionaries containing the number of pixels in
        each segment mapping.

    Outputs:
    (total - p) -- Pixel error.
    L -- Number of ground truth segments that have a mapping in the inferred mask
    best -- True mapping as dictionary.

    """
    p = 0
    #L = len(setcounts)

    total = 0
    L_sets = set()

    best = dict()

    for i_key in setcounts:
        my_mx = 0
        mx_key = ''
        for g_key in setcounts[i_key]:
            total += setcounts[i_key][g_key]  # add to total pixel count
            if setcounts[i_key][g_key] > my_mx:
                my_mx = setcounts[i_key][g_key]
                # mx_key = i_key
                mx_key = g_key  # record mapping with greatest pixel count
        p += my_mx
        # L_sets.add(g_key)
        L_sets.add(mx_key)  # add the g_key we consider to be correct
        # best[i_key] = g_key
        best[i_key] = mx_key  # record "true" mapping
    L = len(L_sets)
    return total - p, L, best


def FF_Option1(inferred, ground_truth):
    """Compute the fitness for an individual.

    Takes in two images and compares
    them according to the equation (p + 2)^log(|m - n| + 2), where p is the pixel
    error, m is the number of segments in the inferred mask, and n is the number
    of segments in the ground truth mask.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    ground_truth -- Ground truth segmentation mask for training image.

    Outputs:
    error -- fitness value as float
    best -- true mapping as dictionary

    """
    # makes sure images are in grayscale
    if len(inferred.shape) > 2:
        inferred = color.rgb2gray(inferred)
    if len(ground_truth.shape) > 2:  # comment out
        ground_truth = color.rgb2gray(ground_truth)  # comment out

    # Replace with function to output p an L
    # p - number of pixels not correcly mapped
    # L - Number of correctly mapped sets
    setcounts, m, n = countMatches(inferred, ground_truth)

    # print(setcounts)
    p, L, _ = countsets(setcounts)
    error = L * (p + 2)

    return [error, ]


def FF_Option2a(inferred, ground_truth):
    """Compute the fitness for an individual.

    Takes in two images and compares
    them according to the equation (p + 2)^log(|m - n| + 2), where p is the pixel
    error, m is the number of segments in the inferred mask, and n is the number
    of segments in the ground truth mask.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    ground_truth -- Ground truth segmentation mask for training image.

    Outputs:
    error -- fitness value as float
    best -- true mapping as dictionary

    """
    # makes sure images are in grayscale
    if len(inferred.shape) > 2:
        inferred = color.rgb2gray(inferred)
    if len(ground_truth.shape) > 2:  # comment out
        ground_truth = color.rgb2gray(ground_truth)  # comment out

    # Replace with function to output p an L
    # p - number of pixels not correcly mapped
    # L - Number of correctly mapped sets
    setcounts, m, n = countMatches(inferred, ground_truth)

    # print(setcounts)
    p, L, _ = countsets(setcounts)

    error = (p + 2)**(np.abs(m - n))

    return [error, ]


def FF_Option2b(inferred, ground_truth):
    """Compute the fitness for an individual.

    Takes in two images and compares
    them according to the equation (p + 2)^log(|m - n| + 2), where p is the pixel
    error, m is the number of segments in the inferred mask, and n is the number
    of segments in the ground truth mask.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    ground_truth -- Ground truth segmentation mask for training image.

    Outputs:
    error -- fitness value as float
    best -- true mapping as dictionary

    """
    # makes sure images are in grayscale
    if len(inferred.shape) > 2:
        inferred = color.rgb2gray(inferred)
    if len(ground_truth.shape) > 2:  # comment out
        ground_truth = color.rgb2gray(ground_truth)  # comment out

    # Replace with function to output p an L
    # p - number of pixels not correcly mapped
    # L - Number of correctly mapped sets
    setcounts, m, n = countMatches(inferred, ground_truth)

    # print(setcounts)
    p, L, _ = countsets(setcounts)

    error = (p + 2)**(np.abs(m - n) + 1)

    return [error, ]


def FitnessFunction_old(inferred, ground_truth):
    """Compute the fitness for an individual.

    Takes in two images and compares
    them according to the equation (p + 2)^log(|m - n| + 2), where p is the pixel
    error, m is the number of segments in the inferred mask, and n is the number
    of segments in the ground truth mask.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    ground_truth -- Ground truth segmentation mask for training image.

    Outputs:
    error -- fitness value as float
    best -- true mapping as dictionary

    """
    # makes sure images are in grayscale
    if len(inferred.shape) > 2:
        inferred = color.rgb2gray(inferred)
    if len(ground_truth.shape) > 2:  # comment out
        ground_truth = color.rgb2gray(ground_truth)  # comment out

    # Replace with function to output p an L
    # p - number of pixels not correcly mapped
    # L - Number of correctly mapped sets
    setcounts, m, n = countMatches(inferred, ground_truth)

    # print(setcounts)
    p, L, _ = countsets(setcounts)

    error = (p + 2) ** np.log(abs(m - n) + 2)  # / (L >= n)
    # error = (repeat_count + 2)**(abs(m - n)+1)
    # print(f"TESTING - L={L} < n={n} p={p} m={m} error = {error} ")
    if (L < n) or error <= 0 or error == np.inf or error == np.nan:
        print(
            f"WARNING: Fitness bounds exceeded, using Maxsize - {L} < {n} or {error} <= 0 or {error} == np.inf or {error} == np.nan:"
        )
        error = sys.maxsize
        # print(error)
    return [error, ]


def FF_Normal(inferred, ground_truth):
    """Compute the fitness for an individual.

    Takes in two images and compares
    them according to the equation (p + 2)^log(|m - n| + 2), where p is the pixel
    error, m is the number of segments in the inferred mask, and n is the number
    of segments in the ground truth mask.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    ground_truth -- Ground truth segmentation mask for training image.

    Outputs:
    error -- fitness value as float
    best -- true mapping as dictionary

    """
    # makes sure images are in grayscale
    if len(inferred.shape) > 2:
        inferred = color.rgb2gray(inferred)
    if len(ground_truth.shape) > 2:  # comment out
        ground_truth = color.rgb2gray(ground_truth)  # comment out

    tot_num_pixels = ground_truth.shape[0] * ground_truth.shape[1]
    # Replace with function to output p an L
    # p - number of pixels not correcly mapped
    # L - Number of correctly mapped sets
    setcounts, m, n = countMatches(inferred, ground_truth)

    # print(setcounts)
    p, L, _ = countsets(setcounts)

    # Normalize:
    p = p / tot_num_pixels
    m = m / tot_num_pixels
    n = n / tot_num_pixels

    error = (p + 2) ** np.log(abs(m - n) + 2)  # / (L >= n)
    # error = (repeat_count + 2)**(abs(m - n)+1)
    # print(f"TESTING - L={L} < n={n} p={p} m={m} error = {error} ")
    if (L < n) or error <= 0 or error == np.inf or error == np.nan:
        error = sys.maxsize
        # print(error)
    return [error, ]


def FF_ML2DHD(inferred, ground_truth):
    """Compute the fitness for an individual.

    Takes in two images and compares
    them according to the equation (p + 2)^log(|m - n| + 2), where p is the pixel
     error, m is the number of segments in the inferred mask, and n is the number
       of segments in the ground truth mask.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    ground_truth -- Ground truth segmentation mask for training image.

    Outputs:
    error -- fitness value as float
    best -- true mapping as dictionary

    """
    # makes sure images are in grayscale
    if len(inferred.shape) > 2:
        inferred = color.rgb2gray(inferred)
    if len(ground_truth.shape) > 2:  # comment out
        ground_truth = color.rgb2gray(ground_truth)  # comment out

    tot_num_pixels = ground_truth.shape[0] * ground_truth.shape[1]

    M = ground_truth.shape[0]
    N = ground_truth.shape[1]

    # Replace with function to output p an L
    # p - number of pixels not correcly mapped
    # L - Number of correctly mapped sets
    setcounts, m, n = countMatches(inferred, ground_truth)

    # print(setcounts)
    p, L, _ = countsets(setcounts)

    error = (p + np.abs(n - m)) / (N * M)

    return [error, n, m]


def FF_Hamming(inferred, ground_truth):
    """Compute the fitness for an individual.

    Takes in two images and compares
    them according to the equation (p + 2)^log(|m - n| + 2), where p is the pixel
    error, m is the number of segments in the inferred mask, and n is the number
    of segments in the ground truth mask.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    ground_truth -- Ground truth segmentation mask for training image.

    Outputs:
    error -- fitness value as float
    best -- true mapping as dictionary

    """
    # makes sure images are in grayscale
    if len(inferred.shape) > 2:
        inferred = color.rgb2gray(inferred)
    if len(ground_truth.shape) > 2:  # comment out
        ground_truth = color.rgb2gray(ground_truth)  # comment out

    M = ground_truth.shape[0]
    N = ground_truth.shape[1]

    hamming = 0
    for r in range(M):
        for c in range(N):
            hamming += ground_truth[r, c] != inferred[r, c]

    return [hamming / (M * N), ]


def FF_Gamma(inferred, ground_truth):
    """Compute the fitness for an individual.

    Takes in two images and compares
    them according to the equation (p + 2)^log(|m - n| + 2), where p is the pixel
    error, m is the number of segments in the inferred mask, and n is the number
    of segments in the ground truth mask.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    ground_truth -- Ground truth segmentation mask for training image.

    Outputs:
    error -- fitness value as float
    best -- true mapping as dictionary

    """
    # makes sure images are in grayscale
    if len(inferred.shape) > 2:
        inferred = color.rgb2gray(inferred)
    if len(ground_truth.shape) > 2:  # comment out
        ground_truth = color.rgb2gray(ground_truth)  # comment out

    inferred = inferred > 0
    ground_truth = ground_truth > 0

    M = ground_truth.shape[0]
    N = ground_truth.shape[1]

    def f(u, v): return u + v - (2 * u * v)
    hamming = 0
    for r in range(M):
        for c in range(N):
            hamming += ground_truth[r, c] != inferred[r, c]

    gamma = np.abs(1 - (2 * hamming / (M * N)))

    return [1 - gamma, ]


def FF_ML2DHD(inferred, ground_truth):
    # TODO: Rename, figure out meaning of name
    """Compute the fitness for an individual.

    Takes in two images and compares
    them according to the equation (p + 2)^log(|m - n| + 2), where p is the pixel
    error, m is the number of segments in the inferred mask, and n is the number
    of segments in the ground truth mask.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    ground_truth -- Ground truth segmentation mask for training image.

    Outputs:
    error -- fitness value as float
    best -- true mapping as dictionary

    """
    # makes sure images are in grayscale
    if len(inferred.shape) > 2:
        inferred = color.rgb2gray(inferred)
    if len(ground_truth.shape) > 2:  # comment out
        ground_truth = color.rgb2gray(ground_truth)  # comment out

    TP = ground_truth.shape[0] * ground_truth.shape[1]

    M = ground_truth.shape[0]
    N = ground_truth.shape[1]

    # Replace with function to output p an L
    # p - number of pixels not correcly mapped
    # L - Number of correctly mapped sets
    setcounts, m, n = countMatches(inferred, ground_truth)

    # print(setcounts)
    p, L, _ = countsets(setcounts)

    error = p / TP + np.abs(n - m) / TP

    return [error, n, m]


def FF_ML2DHD_V2(inferred, ground_truth):
    # TODO: Rename, figure out meaning of name
    """Compute the fitness for an individual.

    Takes in two images and compares
    them according to the equation (p + 2)^log(|m - n| + 2), where p is the pixel
    error, m is the number of segments in the inferred mask, and n is the number
    of segments in the ground truth mask.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    ground_truth -- Ground truth segmentation mask for training image.

    Outputs:
    error -- fitness value as float
    best -- true mapping as dictionary

    """
    # makes sure images are in grayscale
    if len(inferred.shape) > 2:
        inferred = color.rgb2gray(inferred)
    if len(ground_truth.shape) > 2:  # comment out
        ground_truth = color.rgb2gray(ground_truth)  # comment out

    TP = ground_truth.shape[0] * ground_truth.shape[1]

    M = ground_truth.shape[0]
    N = ground_truth.shape[1]

    # Replace with function to output p an L
    # p - number of pixels not correcly mapped
    # L - Number of correctly mapped sets
    setcounts, m, n = countMatches(inferred, ground_truth)

    # print(setcounts)
    p, L, best = countsets(setcounts)

    test = set()
    for key in best:
        test.add(best[key])

    if len(test) == 1:
        # Trivial Solution
        #print(f"trivial solution")
        #ERROR - Length of test is only 1
        error = 1.5
    else:
        error = (p / TP + np.abs(n - m) / (n + m)
                 )**(1 - np.abs(n - m) / (n + m))

    return [error, n, m]


def FitnessFunction(inferred, ground_truth):
    """Return fitness function result from inferred and ground_truth.

    Keyword arguments:
    inferred -- Resulting segmentation mask from individual.
    ground_truth -- Ground truth segmentation mask for training image.

    Outputs:
    error -- fitness value as float
    best -- true mapping as dictionary
    """
    return FF_ML2DHD_V2(inferred, ground_truth)

#data_arr to store inferred and ground_truth as matrices in numpy arrays
def multi_value_ff(data):
    fitness_values_arr = np.arange(0)
    for i in range(len(data)):
        fitness_value = FitnessFunction(data[i][-1], data.gtruth[i])[0]
        fitness_values_arr = np.append(fitness_values_arr, fitness_value)
    mean_fitness_value = np.mean(fitness_values_arr)
    return mean_fitness_value


class segment_fitness(algorithm):
    """Contains functions to return result of fitness function.

    and run segmentation algorithm
    """

    def __init__(self, paramlist=None):
        """Generate algorithm params from parameter list."""
        super(segment_fitness, self).__init__(paramlist)

    def evaluate(self, data):
        """Return result of fitness function with image and its ground truth.

        Keyword arguments:
        mask -- the given image
        gmask -- the ground truth mask image
        """
        return multi_value_ff(data)

    def pipe(self, data):
        """Run segmentation algorithm to get inferred mask."""
        data.fitness = multi_value_ff(data)
        return data

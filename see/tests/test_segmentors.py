"""This runs unit tests for functions that can be found in Segmentors.py."""
import numpy as np
from skimage import segmentation
from see import Segmentors
from see import Segment_Fitness as SSM
from see.base_classes import pipedata

# Define toy rgb and grayscale images used for testing below
TEST_IM_COLOR = np.zeros((20, 20, 3))
TEST_IM_COLOR[4:10, 4:10, :] = 1.0
TEST_IM_GRAY = TEST_IM_COLOR[:, :, 0]


# TODO: Need new print best algorithm tests.
# I don't like this because our output string will be differnet each time.
# def test_print_best_algorithm_code():
#     """Unit test for print_best_algorithm_code function.
#      Checks function output matches method contents it's printing."""
#     individual = ['FB', 0, 0, 984, 0.09, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
#      (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
#     print_statement = "multichannel = False\n\
# if len(img.shape) > 2:\n\
#     multichannel = True\n\
# output = skimage.segmentation.felzenszwalb(\n\
#     img,\n\
#     984,\n\
#     0.09,\n\
#     92,\n\
#     multichannel=multichannel,\n\
# )\n"
#     assert Segmentors.print_best_algorithm_code(individual) == print_statement

def test_run_algo():
    """Unit test for runAlgo function.
     Checks to see if the output is what it's supposed to be in this case."""
    individual = Segmentors.segmentor()
    data = pipedata()
    data.append([TEST_IM_COLOR])
    data.gtruth.append(TEST_IM_COLOR[:, :, 0])
    individual.runAlgo(data)


def test_parameter_len():
    """Unit test for parameters function. Checks formatting of parameter."""
    param = Segmentors.segmentor().params
    assert len(param) > 1


# TODO Add colorthreshold test.

def test_felzenszwalb():
    """Unit test for Felzenszwalb method. Checks if evaluate function output\
     is the same as manually running the skimage function."""
    fb1 = Segmentors.Felzenszwalb()

    out1 = fb1.evaluate(TEST_IM_COLOR)
    out2 = segmentation.felzenszwalb(
        TEST_IM_COLOR, 984, 0.09, 92, channel_axis=True)
    assert out1.all() == out2.all()

    out1 = fb1.evaluate(TEST_IM_GRAY)
    out2 = segmentation.felzenszwalb(
        TEST_IM_GRAY, 984, 0.09, 92, channel_axis=True)
    assert out1.all() == out2.all()


def test_slic():
    """Unit test for Slic method. Checks if evaluate function output\
     is the same as manually running the skimage function."""
    sc1 = Segmentors.Slic()
    assert sc1.evaluate(TEST_IM_COLOR).all() == segmentation.slic(
        TEST_IM_COLOR, n_segments=5, compactness=5, max_num_iter=3,
        sigma=5, convert2lab=True, ).all()
    assert sc1.evaluate(TEST_IM_GRAY).all() == segmentation.slic(
        TEST_IM_GRAY, n_segments=5, compactness=5, max_num_iter=3,
        sigma=5, convert2lab=True, channel_axis=None).all()

# def test_QuickShift():
#     """Unit test for QuickShift method. Checks if evaluate function output\
#      is the same as manually running the skimage function."""
#     qs1 = Segmentors.QuickShift()
#     assert qs1.evaluate(TEST_IM_COLOR).all() == segmentation.quickshift(\
#                 TEST_IM_COLOR, ratio=2, kernel_size=5, max_dist=60, sigma=5, random_seed=1).all()
#     assert qs1.evaluate(TEST_IM_GRAY).all() == segmentation.quickshift(color.gray2rgb(\
#                 TEST_IM_GRAY), ratio=2, kernel_size=5, max_dist=60, sigma=5, random_seed=1).all()

# def test_Watershed():
#     """Unit test for Watershed method. Checks if evaluate function output\
#      is the same as manually running the skimage function."""
#     ws1 = Segmentors.Watershed()
#     assert ws1.evaluate(TEST_IM_COLOR).all() == segmentation.watershed(\
#                 TEST_IM_COLOR, markers=None, compactness=2.0).all()

# def test_Chan_Vese():
#     """Unit test for Chan_Vese method. Checks if evaluate function output\
#      is the same as manually running the skimage function."""
#     cv1 = Segmentors.Chan_Vese()
#     assert cv1.evaluate(TEST_IM_COLOR).all() == segmentation.chan_vese(\
#                 color.rgb2gray(TEST_IM_COLOR), mu=2.0, lambda1=10, \
#                 lambda2=20, tol=0.001, max_iter=10, dt=0.10).all()
#     assert cv1.evaluate(TEST_IM_GRAY).all() == segmentation.chan_vese(\
#                 TEST_IM_GRAY, mu=2.0, lambda1=10, \
#                 lambda2=20, tol=0.001, max_iter=10, dt=0.10).all()


# def test_Morphological_Chan_Vese():
#     """Unit test for Morphological_Chan_Vese method. Checks if evaluate function output\
#      is the same as manually running the skimage function."""
#     mcv1 = Segmentors.Morphological_Chan_Vese()
#     assert mcv1.evaluate(TEST_IM_COLOR).all() == segmentation.morphological_chan_vese(\
#                 color.rgb2gray(TEST_IM_COLOR), iterations=10, init_level_set="checkerboard", \
#                 smoothing=10, lambda1=10, lambda2=20).all()
#     assert mcv1.evaluate(TEST_IM_GRAY).all() == segmentation.morphological_chan_vese(\
#                 TEST_IM_GRAY, iterations=10, init_level_set="checkerboard", \
#                 smoothing=10, lambda1=10, lambda2=20).all()

# def test_MorphGeodesicActiveContour():
#     """Unit test for MorphGeodesicActiveContour method. Checks if evaluate function output\
#      is the same as manually running the skimage function."""
#     ac1 = Segmentors.MorphGeodesicActiveContour()
#     assert ac1.evaluate(TEST_IM_COLOR).all() == segmentation.morphological_geodesic_active_contour(\
#                 segmentation.inverse_gaussian_gradient(color.rgb2gray(TEST_IM_COLOR), 0.2, 0.3),\
#                 iterations=10, init_level_set='checkerboard', smoothing=5, threshold='auto',\
#                 balloon=10).all()
#     assert ac1.evaluate(TEST_IM_GRAY).all() == segmentation.morphological_geodesic_active_contour(\
#                 segmentation.inverse_gaussian_gradient(TEST_IM_GRAY, 0.2, 0.3), iterations=10,\
#                 init_level_set='checkerboard', smoothing=5, threshold='auto', balloon=10).all()

def test_count_matches():
    """Unit test for countMatches function. Checks output is as
     expected for a variety of extreme cases."""
    # create test image
    ground_truth = np.zeros((20, 20))
    ground_truth[4:10, 4:10] = 1
    inferred = np.zeros((20, 20))
    inferred[4:10, 4:6] = 1
    inferred[4:10, 6:10] = 2
    assert SSM.countMatches(inferred, ground_truth) ==\
        ({0.0: {0.0: 364}, 1.0: {1.0: 12}, 2.0: {1.0: 24}}, 3, 2)

    inferred = np.zeros((20, 20))
    inferred[4:10, 3:6] = 1
    inferred[4:10, 6:10] = 2
    assert SSM.countMatches(inferred, ground_truth) ==\
        ({0.0: {0.0: 358}, 1.0: {0.0: 6, 1.0: 12}, 2.0: {1.0: 24}}, 3, 2)

    inferred = np.zeros((20, 20))
    inferred[4:10, 3:6] = 1
    inferred[4:10, 6:10] = 2
    inferred[3:5, 3:6] = 3
    assert SSM.countMatches(inferred, ground_truth) ==\
        ({0.0: {0.0: 355}, 3.0: {0.0: 4, 1.0: 2},
         2.0: {1.0: 24}, 1.0: {0.0: 5, 1.0: 10}}, 4, 2)

    inferred = np.zeros((20, 20))
    assert SSM.countMatches(inferred, ground_truth) == (
        {0.0: {0.0: 364, 1.0: 36}}, 1, 2)

    inferred = np.zeros((20, 20))
    inferred[1:19, 1:19] = 1
    assert SSM.countMatches(inferred, ground_truth) ==\
        ({0.0: {0.0: 76}, 1.0: {0.0: 288, 1.0: 36}}, 2, 2)


def test_countsets():
    """Unit test for countsets function. Checks output is as
     expected for a variety of extreme cases."""
    assert SSM.countsets({0.0: {0.0: 364}, 1.0: {1.0: 12}, 2.0: {1.0: 24}}) ==\
        (0, 2, {0.0: 0.0, 1.0: 1.0, 2.0: 1.0})
    assert SSM.countsets({0.0: {0.0: 358}, 1.0: {0.0: 6, 1.0: 12}, 2.0: {1.0: 24}}) ==\
        (6, 2, {0.0: 0.0, 1.0: 1.0, 2.0: 1.0})
    assert SSM.countsets({0.0: {0.0: 355}, 3.0: {0.0: 4, 1.0: 2}, 2.0: {1.0: 24},
                          1.0: {0.0: 5, 1.0: 10}}) == (7, 2, {0.0: 0.0, 3.0: 0.0, 2.0: 1.0, 1.0: 1.0})
    assert SSM.countsets({0.0: {0.0: 364, 1.0: 36}}) ==\
        (36, 1, {0.0: 0.0})
    assert SSM.countsets({0.0: {0.0: 76}, 1.0: {0.0: 288, 1.0: 36}}) ==\
        (36, 1, {0.0: 0.0, 1.0: 0.0})

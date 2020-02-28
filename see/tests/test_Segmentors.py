from see import Segmentors
import pytest
import numpy as np
import sys
from skimage import segmentation, color

# Define toy rgb and grayscale images used for testing below
test_im_color = np.zeros((20, 20, 3))
test_im_color[4:10, 4:10, :] = 1
test_im_gray = test_im_color[:, :, 0]

def test_runAlgo():
    individual = ['FB', 0, 0, 984, 0.09, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    assert Segmentors.runAlgo(test_im_color, test_im_color[:, :, 0], individual) == [sys.maxsize]

def test_parameters():
    param = Segmentors.parameters()
    assert param.printparam('min_size') == "min_size=0.0\n\tparameter for felzenszwalb\n\t[i for i in range(0,10000)]\n"

def test_Felzenszwalb():
    FB1 = Segmentors.Felzenszwalb()
    assert FB1.evaluate(test_im_color).all() == segmentation.felzenszwalb(\
            test_im_color, 984, 0.09, 92, multichannel=True).all()
    assert FB1.evaluate(test_im_gray).all() == segmentation.felzenszwalb(\
            test_im_gray, 984, 0.09, 92, multichannel=False).all()

def test_Slic():
    SC1 = Segmentors.Slic()
    assert SC1.evaluate(test_im_color).all() == segmentation.slic(\
                test_im_color, n_segments=5, compactness=5, max_iter=3, \
                sigma=5, convert2lab=True, multichannel=True).all()
    assert SC1.evaluate(test_im_gray).all() == segmentation.slic(\
                test_im_gray, n_segments=5, compactness=5, max_iter=3, \
                sigma=5, convert2lab=True, multichannel=False).all()

def test_QuickShift():
    QS1 = Segmentors.QuickShift()
    assert QS1.evaluate(test_im_color).all() == segmentation.quickshift(\
                test_im_color, ratio=2, kernel_size=5, max_dist=60, sigma=5, random_seed=1).all()
    assert QS1.evaluate(test_im_gray).all() == segmentation.quickshift(\
                color.gray2rgb(test_im_gray), ratio=2, kernel_size=5, max_dist=60, sigma=5, random_seed=1).all()

def test_Watershed():
    WS1 = Segmentors.Watershed()
    assert WS1.evaluate(test_im_color).all() == segmentation.watershed(\
                test_im_color, markers=None, compactness=2.0).all()
    assert WS1.evaluate(test_im_gray).all() == segmentation.watershed(\
                test_im_gray, markers=None, compactness=2.0).all()

def test_Chan_Vese():
    CV1 = Segmentors.Chan_Vese()
    assert CV1.evaluate(test_im_color).all() == segmentation.chan_vese(\
                color.rgb2gray(test_im_color), mu=2.0, lambda1=10, \
                lambda2=20, tol=0.001, max_iter=10, dt=0.10).all()
    assert CV1.evaluate(test_im_gray).all() == segmentation.chan_vese(\
                test_im_gray, mu=2.0, lambda1=10, \
                lambda2=20, tol=0.001, max_iter=10, dt=0.10).all()


def test_Morphological_Chan_Vese():
    MCV1 = Segmentors.Morphological_Chan_Vese()
    assert MCV1.evaluate(test_im_color).all() == segmentation.morphological_chan_vese(\
                color.rgb2gray(test_im_color), iterations=10, init_level_set="checkerboard", \
                smoothing=10, lambda1=10, lambda2=20).all()
    assert MCV1.evaluate(test_im_gray).all() == segmentation.morphological_chan_vese(\
                test_im_gray, iterations=10, init_level_set="checkerboard", \
                smoothing=10, lambda1=10, lambda2=20).all()

def test_MorphGeodesicActiveContour():
    AC1 = Segmentors.MorphGeodesicActiveContour()
    assert AC1.evaluate(test_im_color).all() == segmentation.morphological_geodesic_active_contour(\
                segmentation.inverse_gaussian_gradient(color.rgb2gray(test_im_color), 0.2, 0.3), \
                iterations=10, init_level_set='checkerboard', smoothing=5, threshold='auto', balloon=10).all()
    assert AC1.evaluate(test_im_gray).all() == segmentation.morphological_geodesic_active_contour(\
                segmentation.inverse_gaussian_gradient(test_im_gray, 0.2, 0.3), \
                iterations=10, init_level_set='checkerboard', smoothing=5, threshold='auto', balloon=10).all()

def test_countMatches():
    # create test image
    groundTruth = np.zeros((20, 20))
    groundTruth[4:10, 4:10] = 1
    inferred = np.zeros((20, 20))
    inferred[4:10, 4:6] = 1
    inferred[4:10, 6:10] = 2
    assert Segmentors.countMatches(inferred, groundTruth) == ({0.0: {0.0: 364}, 1.0: {1.0: 12}, 2.0: {1.0: 24}}, 3, 2)

    inferred = np.zeros((20, 20))
    inferred[4:10, 3:6] = 1
    inferred[4:10, 6:10] = 2
    assert Segmentors.countMatches(inferred, groundTruth) == ({0.0: {0.0: 358}, 1.0: {0.0: 6, 1.0: 12}, 2.0: {1.0: 24}}, 3, 2)

    inferred = np.zeros((20, 20))
    inferred[4:10, 3:6] = 1
    inferred[4:10, 6:10] = 2
    inferred[3:5, 3:6] = 3
    assert Segmentors.countMatches(inferred, groundTruth) == ({0.0: {0.0: 355}, 3.0: {0.0: 4, 1.0: 2}, 2.0: {1.0: 24}, 1.0: {0.0: 5, 1.0: 10}}, 4, 2)

    inferred = np.zeros((20, 20))
    assert Segmentors.countMatches(inferred, groundTruth) == ({0.0: {0.0: 364, 1.0: 36}}, 1, 2)

    inferred = np.zeros((20, 20))
    inferred[1:19, 1:19] = 1
    assert Segmentors.countMatches(inferred, groundTruth) == ({0.0: {0.0: 76}, 1.0: {0.0: 288, 1.0: 36}}, 2, 2)


def test_countsets():
    assert Segmentors.countsets({0.0: {0.0: 364}, 1.0: {1.0: 12}, 2.0: {1.0: 24}}) == (0, 2, {0.0: 0.0, 1.0: 1.0, 2.0: 1.0})
    assert Segmentors.countsets({0.0: {0.0: 358}, 1.0: {0.0: 6, 1.0: 12}, 2.0: {1.0: 24}}) == (6, 2, {0.0: 0.0, 1.0: 1.0, 2.0: 1.0})
    assert Segmentors.countsets({0.0: {0.0: 355}, 3.0: {0.0: 4, 1.0: 2}, 2.0: {1.0: 24}, 1.0: {0.0: 5, 1.0: 10}}) == (7, 2, {0.0: 0.0, 3.0: 0.0, 2.0: 1.0, 1.0: 1.0})
    assert Segmentors.countsets({0.0: {0.0: 364, 1.0: 36}}) == (36, 1, {0.0: 0.0})
    assert Segmentors.countsets({0.0: {0.0: 76}, 1.0: {0.0: 288, 1.0: 36}}) == (36, 1, {0.0: 0.0, 1.0: 0.0})


def test_FitnessFunction():
    # create test image
    groundTruth = np.zeros((20, 20))
    groundTruth[4:10, 4:10] = 1
    inferred = np.zeros((20, 20))
    inferred[4:10, 4:6] = 1
    inferred[4:10, 6:10] = 2
    assert Segmentors.FitnessFunction(inferred, groundTruth) == [2 ** np.log(3),]

    inferred = np.zeros((20, 20))
    inferred[4:10, 3:6] = 1
    inferred[4:10, 6:10] = 2
    assert Segmentors.FitnessFunction(inferred, groundTruth) == [8 ** np.log(3),]

    inferred = np.zeros((20, 20))
    inferred[4:10, 3:6] = 1
    inferred[4:10, 6:10] = 2
    inferred[3:5, 3:6] = 3
    assert Segmentors.FitnessFunction(inferred, groundTruth) == [9 ** np.log(4),]

    inferred = np.zeros((20, 20))
    assert Segmentors.FitnessFunction(inferred, groundTruth) == [sys.maxsize,]

    inferred = np.arange(400).reshape(groundTruth.shape)
    assert Segmentors.FitnessFunction(inferred, groundTruth) == [2 ** np.log(400),]

    inferred = np.zeros((20, 20))
    inferred[1:19, 1:19] = 1
    assert Segmentors.FitnessFunction(inferred, groundTruth) == [sys.maxsize,]
    
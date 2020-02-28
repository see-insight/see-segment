from see import Segmentors
from see import GeneticSearch
import pytest
import numpy as np

def test_printBestAlgorithmCode():
    individual = ['FB', 0, 0, 984, 0.09, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    print_statement ="multichannel = False\n\
if len(img.shape) > 2:\n\
    multichannel = True\n\
output = skimage.segmentation.felzenszwalb(\n\
    img,\n\
    984,\n\
    0.09,\n\
    92,\n\
    multichannel=multichannel,\n\
)\n"
    assert GeneticSearch.printBestAlgorithmCode(individual) == print_statement

def test_twoPointCopy():
    np1 = ['FB', 0, 0, 984, 0.09, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    np2 = ['CT', 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 0, 0, 0, 0, (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    new_np1, new_np2 = GeneticSearch.twoPointCopy(np1, np2, True)
    assert new_np1 == ['FB', 0, 0, 0, 0.09, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1, 2), 0, 'checkerboard', 'checkerboard', 0, 0, 0, 0, 0, 0]
    assert new_np2 == ['CT', 0, 0, 984, 0, 0, 0, 0, 8, 10, 12, 0, 0, 0, 0, (1, 2), 0, 'checkerboard', 'checkerboard', 0, 0, 0, 0, 0, 0]

def test_skimageCrossRandom():
    np1 = ['FB', 0, 0, 984, 0.09, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    np2 = ['CT', 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 0, 0, 0, 0, (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    new_np1, new_np2 = GeneticSearch.skimageCrossRandom(np1, np2)
    assert new_np1 == ['FB', 0, 0, 984, 0.09, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1, 2), 0, 'checkerboard', 'checkerboard', 0, 0, 0, 0, 0, 0]
    assert new_np2 == ['CT', 0, 0, 0, 0, 92, 0, 0, 8, 10, 12, 0, 0, 0, 0, (1, 2), 0, 'checkerboard', 'checkerboard', 0, 0, 0, 0, 0, 0]

def test_mutate():
    copyChild = ['FB', 0, 0, 984, 0.09, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    AllVals = []
    p = Segmentors.parameters()
    for key in p.pkeys:
        AllVals.append(eval(p.ranges[key]))
    assert type(GeneticSearch.mutate(copyChild, AllVals, 0.5, True)) == list
    assert GeneticSearch.mutate(copyChild, AllVals, 0.5, True) == ['FB', 1390, 0.173, 984, 0.09, 9927, 587, 0, 0.55, 0, 0, 0, 0, 1000, 0, (1, 2), 0, 'disk', 'checkerboard', 9, 2907, -47, (0.0, 0.0, 0.0), 0, 0]

def test_makeToolbox():
    assert GeneticSearch.makeToolbox(10).population.keywords['n'] == 10

def test_newpopulation():
    img = np.zeros((20, 20, 3))
    img[4:10, 4:10, :] = 1
    mask = img[:, :, 0]
    ee = GeneticSearch.Evolver(img, mask)
    assert type(ee.tool.population()) == list
    assert len(ee.tool.population()) == 10

def test_popfitness():
    img = np.zeros((20, 20, 3))
    img[4:10, 4:10, :] = 1
    mask = img[:, :, 0]
    ee = GeneticSearch.Evolver(img, mask)
    fits, tpop = ee.popfitness(ee.tool.population())
    assert type(fits) == list
    assert len(fits) == 10
    assert type(tpop) == list
    assert len(tpop) == 10

def test_mutate():
    img = np.zeros((20, 20, 3))
    img[4:10, 4:10, :] = 1
    mask = img[:, :, 0]
    ee = GeneticSearch.Evolver(img, mask)
    tpop = ee.mutate(ee.tool.population())
    assert type(tpop) == list
    assert len(tpop) == 10

def test_nextgen():
    img = np.zeros((20, 20, 3))
    img[4:10, 4:10, :] = 1
    mask = img[:, :, 0]
    ee = GeneticSearch.Evolver(img, mask)
    pop = ee.tool.population()
    tpop = ee.mutate(pop)
    assert type(tpop) == list
    assert len(tpop) ==  10
    assert tpop != pop

def test_run():
    img = np.zeros((20, 20, 3))
    img[4:10, 4:10, :] = 1
    mask = img[:, :, 0]
    ee = GeneticSearch.Evolver(img, mask)
    start_pop = ee.tool.population()
    final_pop = ee.run()
    assert type(final_pop) == list
    assert len(final_pop) ==  10
    assert final_pop != start_pop

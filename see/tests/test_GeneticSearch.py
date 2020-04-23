"""This runs unit tests for functions that can be found in GeneticSearch.py."""
import pytest
import numpy as np
from see import Segmentors
from see import GeneticSearch

def test_print_best_algorithm_code():
    """Unit test for print_best_algorithm_code function.
     Checks function output matches method contents it's printing."""
    individual = ['FB', 0, 0, 984, 0.09, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
     (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    print_statement = "multichannel = False\n\
if len(img.shape) > 2:\n\
    multichannel = True\n\
output = skimage.segmentation.felzenszwalb(\n\
    img,\n\
    984,\n\
    0.09,\n\
    92,\n\
    multichannel=multichannel,\n\
)\n"
    assert GeneticSearch.print_best_algorithm_code(individual) == print_statement

def test_twoPointCopy():
    """Unit test for twoPointCopy function. Checks test individuals to see
     if copy took place successfully."""
    np1 = ['FB', 0, 0, 984, 0.09, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
     (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    np2 = ['CT', 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 0, 0, 0, 0,\
     (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    new_np1, new_np2 = GeneticSearch.twoPointCopy(np1, np2, True)
    assert new_np1 == ['FB', 0, 0, 984, 0.09, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
     (1, 2), 0, 'checkerboard', 'checkerboard', 0, 0, 0, 0, 0, 0]
    assert new_np2 == ['CT', 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 0, 0, 0, 0,\
     (1, 2), 0, 'checkerboard', 'checkerboard', 0, 0, 0, 0, 0, 0]

def test_skimageCrossRandom():
    """Unit test for skimageCrossRandom function. Checks test individuals to see if crossover
     took place successfully."""
    np1 = ['FB', 0, 0, 984, 0.09, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
     (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    np2 = ['CT', 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 0, 0, 0, 0,\
     (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    new_np1, new_np2 = GeneticSearch.skimageCrossRandom(np1, np2)
    assert new_np1 == ['FB', 0, 0, 984, 0.09, 92, 0, 0, 8, 0, 0, 0, 0, 0, 0,\
     (1, 2), 0, 'checkerboard', 'checkerboard', 0, 0, 0, 0, 0, 0]
    assert new_np2 == ['CT', 0, 0, 0, 0, 0, 0, 0, 0, 10, 12, 0, 0, 0, 0,\
     (1, 2), 0, 'checkerboard', 'checkerboard', 0, 0, 0, 0, 0, 0]

def test_mutate():
    """Unit test for mutate function. Checks output type and checks test individual
     to see if mutation took place successfully."""
    copy_child = ['FB', 0, 0, 984, 0.09, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
     (1, 2), 0, "checkerboard", "checkerboard", 0, 0, 0, 0, 0, 0]
    all_vals = []
    params = Segmentors.parameters()
    for key in params.pkeys:
        all_vals.append(eval(params.ranges[key]))
    assert isinstance(GeneticSearch.mutate(copy_child, all_vals, 0.5, True), list)
    assert GeneticSearch.mutate(copy_child, all_vals, 0.5, True) ==\
     ['FB', 1390, 0.173, 984, 0.09, 9927, 587, 0, 0.55, 0, 0, 0, 0, 1000, 0,\
      (1, 2), 0, 'disk', 'checkerboard', 9, 2907, -47, (0.0, 0.0, 0.0), 0, 0]

def test_makeToolbox():
    """Unit test for makeToolbox function. Checks that a toolbox of the
     correct size was made."""
    assert GeneticSearch.makeToolbox(10).population.keywords['n'] == 10

def test_newpopulation():
    """Unit test for newpopulation function. Checks the type and length of
     the new population."""
    img = np.zeros((20, 20, 3))
    img[4:10, 4:10, :] = 1
    mask = img[:, :, 0]
    evolv = GeneticSearch.Evolver(img, mask)
    assert isinstance(evolv.tool.population(), list)
    assert len(evolv.tool.population()) == 10

def test_popfitness():
    """Unit test for popfitness function. Checks the type and length of the fitness
     values and population."""
    img = np.zeros((20, 20, 3))
    img[4:10, 4:10, :] = 1
    mask = img[:, :, 0]
    evolv = GeneticSearch.Evolver(img, mask)
    fits, tpop = evolv.popfitness(evolv.tool.population())
    assert isinstance(fits, list)
    assert len(fits) == 10
    assert isinstance(tpop, list)
    assert len(tpop) == 10

def test_mutate():
    """Unit test for mutate function. Checks type and length of the
     new population after mutation."""
    img = np.zeros((20, 20, 3))
    img[4:10, 4:10, :] = 1
    mask = img[:, :, 0]
    evolv = GeneticSearch.Evolver(img, mask)
    tpop = evolv.mutate(evolv.tool.population())
    assert isinstance(tpop, list)
    assert len(tpop) == 10

def test_nextgen():
    """Unit test for nextgen function. Checks the type and length of the new population,
     and checks that the population is evolving."""
    img = np.zeros((20, 20, 3))
    img[4:10, 4:10, :] = 1
    mask = img[:, :, 0]
    evolv = GeneticSearch.Evolver(img, mask)
    pop = evolv.tool.population()
    tpop = evolv.mutate(pop)
    assert isinstance(tpop, list)
    assert len(tpop) == 10
    assert tpop != pop

def test_run():
    """Unit test for run function. Checks the type and length of the final population,
     and checks that the population evolved."""
    img = np.zeros((20, 20, 3))
    img[4:10, 4:10, :] = 1
    mask = img[:, :, 0]
    evolv = GeneticSearch.Evolver(img, mask)
    start_pop = evolv.tool.population()
    final_pop = evolv.run()
    assert isinstance(final_pop, list)
    assert len(final_pop) == 10
    assert final_pop != start_pop

import argparse
import sys
import matplotlib.pylab as plt
import imageio
from see import GeneticSearch, Segmentors

def geneticsearch_commandline():
    """Rename Instructor notebook using git and fix all
    student links in files."""
    parser = argparse.ArgumentParser(description='rename notebook')

    parser.add_argument('input_file', help=' input image')
    parser.add_argument('input_mask', help=' input Ground Truthe Mask')

    args = parser.parse_args()

    print('\n\n')
    print(args)
    print('\n\n')

    #Multilabel Array Example
    img = imageio.imread(args.input_file)
    gmask = imageio.imread(args.input_mask)

    #Run the search
    my_evolver = GeneticSearch.Evolver(img, gmask, pop_size=10)
    population = my_evolver.run(ngen=5)
    
    #Get the best algorithm so far
    params = my_evolver.hof[0]
    print('Best Individual:\n', params)
    
    #Generate mask of best so far.
    seg = Segmentors.algoFromParams(params)
    mask = seg.evaluate(img)
    
    imageio.imwrite('./temp_mask.png',mask);
    
    
if __name__ == "__main__":
    print("hello world")
    geneticsearch_commandline()

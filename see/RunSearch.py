import argparse
import sys
import matplotlib.pylab as plt
import imageio
from see import GeneticSearch, Segmentors


def continuous_search(input_file, 
                      input_mask, 
                      best_mask_file="temp_mask.png", 
                      pop_size=10):
    
    img = imageio.imread(input_file)
    gmask = imageio.imread(input_mask)

    #Run the search
    my_evolver = GeneticSearch.Evolver(img, gmask, pop_size=pop_size)
    
    best_fitness=2.0
    iteration = 0
    
    while(best_fitness > 0.0):
        print(f"running {iteration} iteration")
        population = my_evolver.run(ngen=1)

        #Get the best algorithm so far
        params = my_evolver.hof[0]
        print('Best Individual:\n', params)

        #Generate mask of best so far.
        seg = Segmentors.algoFromParams(params)
        mask = seg.evaluate(img)
        
        fitness = Segmentors.FitnessFunction(mask, gmask)[0]
        if (fitness < best_fitness):
            best_fitness = fitness
            print("Iteration {iteration} Finess Improved to {fitness}")
            imageio.imwrite(best_mask_file,mask);
        iteration += 1

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
    
    continuous_search(args.input_file, args.input_mask);
    


#     #Multilabel Array Example
#     img = imageio.imread(args.input_file)
#     gmask = imageio.imread(args.input_mask)

#     #Run the search
#     my_evolver = GeneticSearch.Evolver(img, gmask, pop_size=10)
#     population = my_evolver.run(ngen=5)
    
#     #Get the best algorithm so far
#     params = my_evolver.hof[0]
#     print('Best Individual:\n', params)
    
#     #Generate mask of best so far.
#     seg = Segmentors.algoFromParams(params)
#     mask = seg.evaluate(img)
    
#     imageio.imwrite('./temp_mask.png',mask);
    
    
if __name__ == "__main__":
    print("hello world")
    geneticsearch_commandline()

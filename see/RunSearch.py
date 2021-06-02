import argparse
import sys
import matplotlib.pylab as plt
import imageio
from see import GeneticSearch, Segmentors
import random
from pathlib import Path

from see.Segmentors import segmentor
from see.ColorSpace import colorspace
from see.Workflow import workflow
from see.Segment_Fitness import segment_fitness
from see import base_classes 
from see.git_version import git_version

def read_pop(filename):
    """Read Text output"""
    print(f"Reading in {filename}")
    inlist = []
    fitness = []
    with open(filename,'r') as myfile:
        for line in myfile:
            if (len(line) > 1):
                x,fit,pop = eval(line)
                inlist.append(pop)
                fitness.append(fit)
    return inlist, fitness

def write_vector(fpop_file, outstring):
    """Write Text output"""
    print(f"Writing in {fpop_file}")
    with open(fpop_file, 'a') as myfile:
        myfile.write(f'{outstring}\n')

def continuous_search(input_file, 
                      input_mask, 
                      startfile=None,
                      checkpoint='checkpoint.txt',
                      pop_size=10):
    mydata = base_classes.pipedata()
    mydata.img = imageio.imread(input_file)
    mydata.gmask = imageio.imread(input_mask)


    pname = Path(input_file)
    outfile=pname.parent.joinpath(f"_{pname.stem}.txt")
    mask_file = pname.parent.joinpath(f"{pname.stem}_bestsofar.png")
    
    #TODO: Read this file in and set population first
    workflow.addalgos([colorspace, segmentor, segment_fitness])
    wf = workflow()
    
    my_evolver = GeneticSearch.Evolver(workflow, mydata, pop_size=pop_size)
    population = my_evolver.newpopulation()
    best_fitness=2.0
    if outfile.exists():
        inlist, fitness=read_pop(outfile)
        for fit in fitness:
            if fit < best_fitness:
                best_fitness = fit
        previous_pop = my_evolver.copy_pop_list(inlist)
        if len(previous_pop) > len(population):
            population = previous_pop[:-len(population)]
        else:
            for index, ind in enumerate(previous_pop):
                population[index] = ind
        print(f"######### Done importing previous list {best_fitness}")

    iteration = 0
 
    while(best_fitness > 0.0):
        print(f"running {iteration} iteration")
        population = my_evolver.run(ngen=1,population=population)
            
        #Get the best algorithm so far
        best_so_far = my_evolver.hof[0]
        fitness = best_so_far.fitness.values[0]
        if (fitness < best_fitness):
            best_fitness = fitness
            print(f"\n\n\n\nIteration {iteration} Finess Improved to {fitness}")

            #Generate mask of best so far.
            seg = workflow(paramlist=best_so_far)
            mydata = seg.pipe(mydata)
            imageio.imwrite(mask_file,mydata.mask);
            write_vector(f"{outfile}", f"[{iteration}, {fitness}, {best_so_far}]") 
        iteration += 1

def geneticsearch_commandline():
    """Rename Instructor notebook using git and fix all
    student links in files."""
    parser = argparse.ArgumentParser(description='Run Genetic Search on Workflow')

    parser.add_argument('input_file', help=' input image')
    parser.add_argument('input_mask', help=' input Ground Truthe Mask')
    parser.add_argument('--seed', type=int,default=1, help='Input seed (integer)') 
    args = parser.parse_args()
    
    print('\n\n')
    print(args)
    print('\n\n')
    
    #TODO: add this to the setup.py installer so we include the has in the install. 
    print(f"Current Git HASH: {git_version()}")
    
    random.seed(args.seed)
    
    continuous_search(args.input_file, args.input_mask);

if __name__ == "__main__":
    geneticsearch_commandline()

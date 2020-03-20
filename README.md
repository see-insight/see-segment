# Simple Evolutionary Exploration
---
## Overview  

As technology advances, image data is becoming a common element of research experiments. Studies in everything from self-driving vehicles to plant biology utilize images in some capacity. However, every image analysis problem is different and processing this kind of data and retrieving specific information can be extremely time consuming. Even if a researcher already possesses knowledge in image understanding, it can be time consuming to write and validate a customized solution. Thus, if this process can be automated, a significant amount of researcher time can be recovered. The goal of this project will be to generate and develop an easy-to-use tool that can achieve this automation for image segmentation problems.  

In order to achieve this goal, this project will utilize the power of genetic algorithms. Genetic algorithms apply concepts of evolutionary biology to computational problems. This allows them to have several strengths over other methods including, but not limited to, being simple to implement, having flexible implementation that can adapt easily to different workflows, and having final solutions that are easily interpreted by humans. One way to think of genetic algorithms is as machine learning for machine learning. When presented with an example image, the genetic algorithm will not only find the best image segmentation algorithm to use, but will also find the optimal parameters for that specific algorithm.  

---
## Project Description  

For this project, many of the components necessary already exist. The general genetic algorithm code has already been constructed and has been rewritten into an easy-to-use format. The code exists in a state that is easy to use with Jupyter notebooks as well. However, several key pieces are still missing. Two major parts of genetic algorithms are the fitness function and the genetic representation vector. The fitness function, or measure of error, is mostly finished, although it does still need to be slightly refined. The genetic representation vector, or the search space, currently works but is constructed in a basic way and is subsequently very large. A smaller search space will aid the algorithm in achieving better performance. As part of this project I will aim to refine the fitness function and shrink the overall size of the search space.  

I also hope to clean up the overall code, and package it in a way that makes it easy for others to use. This includes an intuitive user experience and an immediately usable output. Since the genetic algorithm outputs a specific segmentation algorithm with optimal parameters, it is reasonable to believe we could make it output Python code to perform the segmentation process. All the user would have to do is copy the code into a Python script or Jupyter cell, and everything would run automatically from there. 

---
## Running the program
First install all of the necessary packages.

### Dependencies
* python 3.5.3 
  * conda 4.6.14
   * https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
* numpy 1.13.3
  * https://anaconda.org/anaconda/numpy
* scikit-image 0.15.0
 * https://scikit-image.org/docs/dev/install.html 
 * skimage.segmentation
* deap 1.2.2
  * https://anaconda.org/conda-forge/deap
* scoop 0.7.1.1
  * https://scoop.readthedocs.io/en/0.7/install.html
* pillow 6.0.0
  * https://anaconda.org/anaconda/pillow
* pandas 0.24.2
  * https://pandas.pydata.org/pandas-docs/version/0.23.3/install.html
* random 
* math
* copy

### Commands

#### Input arguments
The input arguments are as follows: Seed, Population, Generation, Mutation chance, Mutation Probability and, Crossover chance.

##### SEED
The SEED is used in the quickshift algorithm. The quickshift algorithm accepts a seed that is a C long, however, the length of a C long is platform dependent. Additionally, Seed should be an integer. If you want to set the seed of the whole program, go to after all of the error checking of the input arguments (try/except cases). This should be around line 130.

##### POPULATION
The POPULATION refers to the amount of individuals in the genetic algorithm. This should be a positive integer

##### GENERATIONS
The GENERATIONS refers to the ammount of generations that the algorithm will run for if it does not find a solution This should be a positive integer.

##### MUTATION
The MUTATION variable refers to the chance for each individual to be mutated. As it is a percentage, it is represented as a float from 0 to 1.

##### FLIPPROB
The FLIPPROB variable is very much linked to the mutation chance. It represents the chance that each value associated with the algorithm in question will be mutated. As it is a percentage, this value should be a float from 0 to 1.

##### CROSSOVER
The CROSSOVER variable refers to the crossover chance for any two individuals. The value is a percentage and should be a float from 0 to 1.

###
An example of running the program would be *python main.py 134 10 10 0.5 1 0.5* To run the program regularly.
For parallelization *python -m scoop main.py 134 10 10 0.5 1 0.5*.
For additional commands in scoop, refer to https://scoop.readthedocs.io/en/0.7/usage.html#how-to-launch-scoop-programs

### Features
* In order to change configurations of the program, edit the global variables at the top of the main. 
  * The paths refer to where the input and groundtruth image datasets are
* Prints the best image segmentation found to an image file called "dummy.png"
* Keeps track of the average fitness for each generation and stores it in a text file called "newfile.txt"

### Wanted Features
Primarily, refer to the TODO notes
#### *main.py*
* Make it possible to 'cheat'
  * That is, to seed certain algorithms into the search space. If you already know an algorithm with parameters that may work well for the dataset, you should be able to suggest that algorithm.
* Reduce the memory usage of the program.
  * Around line 80, we make lists of all the paths to all of the images. As we are only looking at one image from each dataset, we only need a single path.
  * We do not always copy the image correctly in every algorithm. As a quick fix, we made a list of a numpy array representation of the images for every individual in the population.
* Currently, the seedpoint for the Flood and Flood_Fill algorithms are selected using a genetic algorithm. It would be helpful if this could be an optional input argument.
  * Perhaps input a range of values?
* Perhaps change the main while loop to Calculate the fitness and then update the population. as opposed to:
* Implement a save state function. This is detailed here: https://deap.readthedocs.io/en/master/tutorials/advanced/checkpoint.html
* Additional error checking for the SEED variable
 * The SEED has to be a C long. A C long is platform dependent, but currentle, it is not being checked for. 
* Redo the overall while loop to:
Calculate Fitness
while
   Update population
   Calculate fitness
   
#### AlgorithmSpace.py in GAHelpers
* Add a color segmentation algorithm
  * Add any image segmentation algorithm following the notes for Adding Additional Algorithms
### GeneticHelp.py in GAHelpers
* The *mutate* function only mutates the values associated with the algorithm that it specifies. This functionality could be useful in the crossover function, *skimageCrossRandom*.
* Our fitness function currently uses the structual similarity index between two images. While this is useful in comparing two images, it is subpar with comparing segmentations. 
  * Change fitness function to evaluate based on the mean squared error
  * Or, change the fitness function to evaluate based on a better method. (I'm not an expert)

## Adding Additional Algorithms
In order to add additional algorithms, it is necessary to edit three different files. Additionally, it is necessary to come up with some information about said algorithm. These are namely:
* A 2-3 character string to represent the algorithm
* The implementation of the algorithm
* The parameters associated with the algorithm
* What channel does this algorithm operate on (e.g. multichannel, grayscale or both)
* If the algorithm returns only a boolean mask of the segmentation
* The range of values that for each parameter

The files are located in GAHelpers. 

### *AlgorithmParams.py*
This files determines all of the parameters for all of the algorithms. At the bottom of the *__init__* constructor, there is a capitalized comment calling to add additional parameters. This is where the parameters of of the new algorithm are written. Additionally, you should also specify an accessor for each parameter. Currently, there is not a use for any of the modifiers, but feel free to create one.

### *AlgorithmHelper.py*
This file provides data and specifications for each algorithm.

#### Constructor
In the *__init__* constructor, there are a few places to edit. The first is in *self.indexes* which is a dictionary. You should add the character code as a key and the indices associated with it as the value. The indices are specified in the *AlgorithmParams.py* file. 
The next part to edit has to do with the channel of the algorithm. If the algorithm runs on grayscale images, append the character code to the *self.GrayAlgos* list. If the algorithm runs on multichannel images, append the character code to the *self.RGBAlgos* list. 
Next, if the algorithm returns a boolean mask, add the character code to the *self.mask* list. Additionally, add the character code to the *self.usedAlgos* list.
It is also important to have the range of values for each parameter. So, for each parameter, with list comprehension, add the range of values for each parameter to *self.PosVals* in the same order that the parameters are listed in *AlgorithmParams.py*

#### *makeToolbox.py*
Make toolbox creates a Toolbox to use with the deap library. It is important to register the parameters to the toolbox. Right before the *func_seq* list, there is a capitalized comment to add more parameters to the toolbox. Follow the format listed to add your parameters. If you want to weight certain values more than others, use *RandHelp.weighted_choice* as opposed to *random.choice*. It is also important to add the the parameters to the *func_seq* list in the same order that they appear in *AlgorithmParams.py*.

### *AlgorithmSpace.py*
The *AlgorithmSpace.py* file contains the implementation for each of the algorithms. To add to this file, look for the ADD NEW ALGORITHMS HERE comment which should be right above the *runAlgo* function and add the implemntation of the algorithm. Any parameters needed can be accesses by *self.params.youAccessor* and the numpy array of the image can be found with *self.params.getImage().getImage()*. 
Finally, add the algorithm to the *switcher* dictionary in *runAlgo*. The key will be the character code of the algorithm while the value will be the implementation of the algorithm. 

#### Notes
* May want to weight QuickShift algorithm as it takes significantly more time to run. Would need to set a timeit function
 * Additionally, weight all algorithms by time taken.
* LabelOne changes already labeled images to binary.
  * Useful if the images are labeled with more than two colors
* May have to clone scikit-image from git, as regular installation installs 0.14 and does not include flood_fill
* We used Python 3.5.3 in order to use the Pillow library. However, with some changes to how we view images, it may be possible to use matplotlib instead. If we use matplotlib, we can use any version of Python.



# TODO List

* Fix Algorithm Defaults
* Speed up fitness
* Seperate Classes into files
* Testing Framework
* Refactor Population Genome
* Make Parallel
  
